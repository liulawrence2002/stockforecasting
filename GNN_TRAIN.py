import torch
import torch.nn.functional as F
from torch.nn import MultiheadAttention
from torch_geometric.data import Data, Dataset, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv
import pandas as pd
import numpy as np
import os
import glob
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import logging
import datetime
from typing import List, Dict, Tuple, Optional, Any
import pickle
import matplotlib.pyplot as plt
import math
import optuna
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(open(os.devnull, 'w')))

DATA_FOLDER = Path("GNN_data")
TIME_AGGREGATE = "15"
TIME_UNIT = "minute"
FILE_SUFFIX = f"{TIME_AGGREGATE}{TIME_UNIT}.csv"
SEQ_LEN = 40
PRED_HORIZON = 2
CORR_THRESHOLD = 0.7
BATCH_SIZE = 16
MAX_EPOCHS = 25
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume']
TARGET_FEATURE = 'Close'
NUM_QUANTILES = 3
QUANTILES = [0.1, 0.5, 0.9]
if 0.5 not in QUANTILES:
    logging.warning("Median quantile 0.5 not found in QUANTILES. RMSE/MAE metrics might not represent median prediction.")
MEDIAN_Q_IDX = QUANTILES.index(0.5) if 0.5 in QUANTILES else -1

N_TRIALS = 3
OPTUNA_TIMEOUT = None
EARLY_STOPPING_PATIENCE = 5
BASE_OUTPUT_DIR = Path("MIDCAPVOL")
HISTOGRAM_LOG_FREQ = 1

def find_data_files(data_folder: Path, suffix: str) -> Dict[str, Path]:
    files = {}
    pattern = f"*_{suffix}"
    for filepath in data_folder.glob(pattern):
        filename = filepath.name
        try:
            ticker = filename.split(f'_{suffix}')[0]
            if ticker:
                files[ticker] = filepath
        except IndexError:
            logging.warning(f"Could not extract ticker from filename: {filename}. Skipping.")
            continue
    logging.info(f"Found {len(files)} data files matching '{pattern}' in '{data_folder}'.")
    if not files:
        logging.warning(f"No files found matching pattern: {pattern}. Ensure datagrab.py ran successfully.")
    return files

def load_and_align_data(ticker_files: Dict[str, Path], features: List[str]) -> pd.DataFrame:
    all_data = {}
    min_date, max_date = pd.Timestamp.max.value, pd.Timestamp.min.value
    for ticker, filepath in ticker_files.items():
        try:
            df = pd.read_csv(filepath, index_col='Timestamp_ms', parse_dates=False,
                             usecols=['Timestamp_ms'] + features,
                             dtype={'Timestamp_ms': np.int64, 'Open': float, 'High': float, 'Low': float,
                                    'Close': float, 'Volume': float})
            if df.empty:
                logging.warning(f"File {filepath} for ticker {ticker} is empty or has no valid data. Skipping.")
                continue
            df = df.astype(float)
            df = df[~df.index.duplicated(keep='first')]
            all_data[ticker] = df
            if not df.empty:
                min_date = min(min_date, df.index.min())
                max_date = max(max_date, df.index.max())
            logging.debug(f"Loaded {ticker}: {len(df)} rows from {df.index.min()} to {df.index.max()}")
        except ValueError as ve:
            logging.error(f"ValueError loading {ticker} from {filepath}. Check dtypes/format: {ve}")
            continue
        except Exception as e:
            logging.error(f"Failed to load or process {ticker} from {filepath}: {e}")
            continue
    if not all_data:
        logging.error("No data loaded successfully.")
        return pd.DataFrame()
    all_indices = set().union(*[set(df.index) for df in all_data.values()])
    full_index = pd.Index(sorted(list(all_indices)), dtype=np.int64)
    logging.info(f"Created combined index using union of timestamps: {len(full_index)} points.")
    reindexed_data = {ticker: df.reindex(full_index) for ticker, df in all_data.items()}
    combined_df = pd.concat(reindexed_data, axis=1, keys=all_data.keys())
    initial_nas = combined_df.isna().sum().sum()
    if initial_nas > 0:
        logging.info(f"Found {initial_nas} missing values after alignment. Applying ffill/bfill.")
        combined_df = combined_df.ffill().bfill()
    else:
        logging.info("No missing values found after alignment.")
    final_nas = combined_df.isna().sum().sum()
    logging.info(f"Aligned data shape: {combined_df.shape}. Handled {initial_nas - final_nas} NaNs. Remaining NaNs: {final_nas}")
    if final_nas > 0:
        logging.warning(f"{final_nas} NaNs remain after filling. Dropping rows with NaNs.")
        rows_before_drop = len(combined_df)
        combined_df = combined_df.dropna(axis=0, how='any')
        rows_after_drop = len(combined_df)
        logging.warning(f"Dropped {rows_before_drop - rows_after_drop} rows with remaining NaNs.")
        logging.info(f"Shape after dropping remaining NaN rows: {combined_df.shape}")
    if combined_df.empty:
        logging.error("DataFrame is empty after NaN handling. Cannot proceed.")
        return pd.DataFrame()
    combined_df.columns = pd.MultiIndex.from_tuples(combined_df.columns, names=['Ticker', 'Feature'])
    combined_df = combined_df.sort_index(axis=1)
    return combined_df

def scale_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Dict[str, StandardScaler]]]:
    scalers: Dict[str, Dict[str, StandardScaler]] = {}
    scaled_df = df.copy()
    tickers = df.columns.get_level_values('Ticker').unique()
    features_to_scale = df.columns.get_level_values('Feature').unique()
    for ticker in tickers:
        scalers[ticker] = {}
        for feature in features_to_scale:
            try:
                data_series = df[(ticker, feature)]
                data_reshaped = data_series.values.reshape(-1, 1)
                if data_reshaped.std() < 1e-8:
                    logging.warning(f"Feature '{feature}' for ticker '{ticker}' has near-zero std deviation. Skipping scaling.")
                    scaled_df[(ticker, feature)] = data_reshaped.flatten()
                    continue
                scaler = StandardScaler()
                scaled_values = scaler.fit_transform(data_reshaped)
                scalers[ticker][feature] = scaler
                scaled_df[(ticker, feature)] = scaled_values.flatten()
            except KeyError:
                logging.warning(f"Column not found for Ticker='{ticker}', Feature='{feature}'. Skipping scaling.")
            except Exception as e:
                logging.error(f"Error scaling Feature='{feature}' for Ticker='{ticker}': {e}")
    logging.info(f"Scaled data for {len(tickers)} tickers and {len(features_to_scale)} features.")
    return scaled_df, scalers

def compute_correlation_edges(data_window: np.ndarray, threshold: float, ticker_list: List[str], features_list: List[str], target_feature: str) -> Tuple[torch.Tensor, torch.Tensor]:
    num_nodes, seq_len, num_features = data_window.shape
    if num_nodes <= 1:
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0, 1), dtype=torch.float)
    try:
        target_idx = features_list.index(target_feature)
    except ValueError:
        logging.error(f"Target feature '{target_feature}' not found in features list: {features_list}. Cannot compute correlations.")
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0, 1), dtype=torch.float)
    target_sequences = data_window[:, :, target_idx]
    df_target = pd.DataFrame(target_sequences.T, columns=ticker_list)
    corr_matrix = df_target.corr().fillna(0).values
    adj = (np.abs(corr_matrix) > threshold).astype(int)
    np.fill_diagonal(adj, 0)
    edge_index = torch.tensor(np.array(np.where(adj)), dtype=torch.long)
    edge_attr_values = corr_matrix[adj == 1]
    edge_attr = torch.tensor(edge_attr_values, dtype=torch.float).unsqueeze(1)
    if edge_index.shape[1] != edge_attr.shape[0]:
        logging.error(f"Mismatch between edge_index count ({edge_index.shape[1]}) and edge_attr count ({edge_attr.shape[0]})")
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0, 1), dtype=torch.float)
    return edge_index, edge_attr

class StockDataset(Dataset):
    def __init__(self, data: pd.DataFrame, tickers: List[str], features: List[str], target_feature: str,
                 seq_len: int, pred_horizon: int, corr_threshold: float):
        super().__init__()
        self.data = data
        self.tickers = tickers
        self.features = features
        self.target_feature = target_feature
        self.seq_len = seq_len
        self.pred_horizon = pred_horizon
        self.corr_threshold = corr_threshold
        self.num_nodes = len(tickers)
        self.ticker_feature_cols = pd.MultiIndex.from_product([self.tickers, self.features], names=['Ticker', 'Feature'])
        self.target_cols = pd.MultiIndex.from_product([self.tickers, [self.target_feature]], names=['Ticker', 'Feature'])
        try:
            self.target_feature_idx_in_list = features.index(target_feature)
        except ValueError:
            logging.error(f"Target feature '{target_feature}' not found in features list: {features}")
            raise
        self.num_samples = len(data) - seq_len - pred_horizon + 1
        if self.num_samples <= 0:
            logging.error(f"Not enough data ({len(data)} rows) for seq_len={seq_len} and pred_horizon={pred_horizon}.")
            self.num_samples = 0
        logging.info(f"Dataset created with {self.num_samples} samples for {self.num_nodes} tickers.")

    def len(self):
        return self.num_samples

    def get(self, idx):
        if idx >= self.num_samples:
            raise IndexError("Index out of bounds")
        start_idx = idx
        end_idx = start_idx + self.seq_len
        target_row_idx = end_idx + self.pred_horizon - 1
        window_data_all_cols = self.data.iloc[start_idx:end_idx]
        x_list = [window_data_all_cols[pd.MultiIndex.from_product([[ticker], self.features], names=['Ticker', 'Feature'])].reindex(columns=pd.MultiIndex.from_product([[ticker], self.features])).values for ticker in self.tickers]
        x = np.stack(x_list, axis=0)
        target_row_data = self.data.iloc[target_row_idx]
        y_values = [target_row_data.get((ticker, self.target_feature), np.nan) for ticker in self.tickers]
        y = np.array(y_values, dtype=np.float32)
        edge_index, edge_attr = compute_correlation_edges(x, self.corr_threshold, self.tickers, self.features, self.target_feature)
        data = Data(x=torch.tensor(x, dtype=torch.float), edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor(y, dtype=torch.float))
        data.num_nodes = self.num_nodes
        return data

class TemporalEncoder(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.lstm = torch.nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)  # -> (num_nodes, input_dim, seq_len)
        x = self.activation(self.conv1(x))
        x = x.permute(0, 2, 1)  # -> (num_nodes, seq_len, hidden_dim)
        x, (hn, cn) = self.lstm(x)
        return hn[0]  # -> (num_nodes, hidden_dim), assuming num_layers=1

class StockGAT(torch.nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, out_dim: int, heads: int):
        super().__init__()
        if node_dim == 0 or out_dim == 0 or heads == 0:
            raise ValueError("GAT dimensions and heads must be > 0")
        if out_dim % heads != 0:
            gat_out_per_head = out_dim // heads
            logging.debug(f"GAT out_dim ({out_dim}) not divisible by heads ({heads}). Using floor division: {gat_out_per_head} per head.")
        else:
            gat_out_per_head = out_dim // heads
        if gat_out_per_head == 0:
            gat_out_per_head = 1
            logging.warning(f"Calculated GAT output per head is 0. Setting to 1.")
        self.gat = GATConv(node_dim, gat_out_per_head, edge_dim=edge_dim, heads=heads, aggr='mean')
        logging.info("StockGAT initialized with aggr='mean'")
        self.activation = torch.nn.ELU()

    def forward(self, x, edge_index, edge_attr):
        if edge_attr is not None and edge_attr.numel() == 0:
            edge_attr = None
        x = self.gat(x, edge_index, edge_attr=edge_attr)
        return self.activation(x)

class CrossAttentionFusion(torch.nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        if embed_dim == 0 or num_heads == 0:
            raise ValueError("Attention embed_dim and num_heads must be > 0")
        if num_heads > 0 and embed_dim % num_heads != 0:
            possible_heads = [h for h in range(num_heads, 0, -1) if embed_dim % h == 0]
            if not possible_heads:
                logging.warning(f"Cannot find suitable num_heads for CrossAttention embed_dim {embed_dim}. Forcing num_heads=1.")
                num_heads = 1
            else:
                num_heads = possible_heads[0]
                logging.warning(f"MultiheadAttention embed_dim ({embed_dim}) not divisible by requested num_heads. Adjusted CrossAttention num_heads to {num_heads}.")
        elif num_heads == 0:
            logging.warning("CrossAttention num_heads calculated as 0. Setting to 1.")
            num_heads = 1
        self.attention = MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True)

    def forward(self, temporal_features, graph_features):
        query = temporal_features.unsqueeze(1)
        key = graph_features.unsqueeze(1)
        value = graph_features.unsqueeze(1)
        attn_output, _ = self.attention(query, key, value)
        fused = temporal_features + attn_output.squeeze(dim=1)
        return fused

class StockForecastGNN(torch.nn.Module):
    def __init__(self, num_features: int, node_hidden_dim: int, gat_heads: int, edge_dim: int, num_quantiles: int):
        super().__init__()
        if node_hidden_dim <= 0 or gat_heads <= 0:
            raise ValueError("node_hidden_dim and gat_heads must be positive integers.")
        self.temporal_encoder = TemporalEncoder(input_dim=num_features, hidden_dim=node_hidden_dim)
        if node_hidden_dim < gat_heads:
            logging.warning(f"node_hidden_dim ({node_hidden_dim}) < gat_heads ({gat_heads}). GAT might be inefficient.")
        gat_out_dim_per_head = node_hidden_dim // gat_heads if gat_heads > 0 else node_hidden_dim
        if gat_heads > 0 and gat_out_dim_per_head == 0:
            gat_out_dim_per_head = 1
            logging.warning(f"Calculated GAT output dim per head is 0. Setting to 1.")
        elif gat_heads <= 0:
            raise ValueError("gat_heads must be positive.")
        gat_internal_out_dim_target = gat_out_dim_per_head * gat_heads
        logging.debug(f"GAT Layer target total out dim: {gat_internal_out_dim_target} ({gat_out_dim_per_head} per head)")
        self.gat = StockGAT(node_dim=node_hidden_dim, edge_dim=edge_dim, out_dim=gat_internal_out_dim_target, heads=gat_heads)
        gat_final_out_dim = gat_internal_out_dim_target
        cross_attn_heads = gat_heads
        self.cross_attn = CrossAttentionFusion(embed_dim=gat_final_out_dim, num_heads=cross_attn_heads)
        tf_d_model = gat_final_out_dim
        tf_nhead = gat_heads
        if tf_nhead > 0 and tf_d_model % tf_nhead != 0:
            possible_heads = [h for h in range(tf_nhead, 0, -1) if tf_d_model % h == 0]
            if not possible_heads:
                logging.warning(f"Cannot find suitable nhead for Transformer d_model {tf_d_model}. Forcing nhead=1.")
                tf_nhead = 1
            else:
                tf_nhead = possible_heads[0]
                logging.warning(f"Transformer d_model ({tf_d_model}) not divisible by requested nhead. Adjusted nhead to {tf_nhead}.")
        elif tf_nhead <= 0:
            logging.warning(f"Transformer head count became zero or negative ({tf_nhead}). Setting to 1.")
            tf_nhead = 1
        self.transformer_layer = torch.nn.TransformerEncoderLayer(d_model=tf_d_model, nhead=tf_nhead, dim_feedforward=tf_d_model * 2, activation='relu', batch_first=True)
        self.output_projection = torch.nn.Linear(tf_d_model, num_quantiles)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        temporal_feats = self.temporal_encoder(x)
        graph_feats = self.gat(temporal_feats, edge_index, edge_attr)
        if temporal_feats.shape[-1] != graph_feats.shape[-1]:
            logging.warning(f"Dim mismatch fusion: T={temporal_feats.shape}, G={graph_feats.shape}. Check model logic.")
            if graph_feats.shape[-1] < temporal_feats.shape[-1]:
                temporal_feats = temporal_feats[:, :graph_feats.shape[-1]]
            elif temporal_feats.shape[-1] < graph_feats.shape[-1]:
                graph_feats = graph_feats[:, :temporal_feats.shape[-1]]
        fused_feats = self.cross_attn(temporal_feats, graph_feats)
        context = self.transformer_layer(fused_feats.unsqueeze(1))
        context = torch.squeeze(context, dim=1)
        quantile_preds = self.output_projection(context)
        return quantile_preds

def quantile_loss(preds: torch.Tensor, targets: torch.Tensor, quantiles: List[float]) -> torch.Tensor:
    targets = targets.unsqueeze(-1).expand_as(preds)
    errors = targets - preds
    q = torch.tensor(quantiles, device=preds.device).unsqueeze(0)
    loss_per_element = torch.max((q - 1) * errors, q * errors)
    mask = ~torch.isnan(targets)
    masked_loss = torch.where(mask, loss_per_element, torch.tensor(0.0, device=preds.device))
    total_loss = torch.sum(masked_loss)
    num_valid_elements = torch.sum(mask)
    if num_valid_elements == 0:
        return torch.tensor(0.0, device=preds.device, requires_grad=True)
    mean_loss = total_loss / num_valid_elements
    return mean_loss

captured_gnn_output = None

def gnn_hook_fn(module, input, output):
    global captured_gnn_output
    if isinstance(output, torch.Tensor):
        captured_gnn_output = output.detach().cpu()
    elif isinstance(output, tuple) and isinstance(output[0], torch.Tensor):
        captured_gnn_output = output[0].detach().cpu()
        logging.debug("Hook captured tuple output from GAT, taking first element.")
    else:
        logging.warning(f"Hook captured unexpected output type from GAT: {type(output)}")
        captured_gnn_output = None

def evaluate(model: torch.nn.Module, loader: DataLoader, criterion: callable, device: torch.device, scalers: Dict[str, Dict[str, StandardScaler]], tickers: List[str], target_feature: str, quantiles: List[float], median_q_idx: int) -> Tuple[float, float, float, Optional[np.ndarray], Optional[np.ndarray]]:
    model.eval()
    total_loss = 0
    num_samples = 0
    all_preds_orig = []
    all_targets_orig = []
    num_nodes = len(tickers)
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="Evaluating", leave=False, disable=True)):
            batch = batch.to(device)
            if batch.num_graphs == 0: continue
            try:
                out = model(batch)
            except Exception as e:
                logging.error(f"[Eval] Error during model forward pass: {e}. Skipping batch {i}.", exc_info=True)
                continue
            target = batch.y
            try:
                loss = criterion(out, target, quantiles)
                if torch.isnan(loss) or torch.isinf(loss):
                    logging.warning(f"[Eval] Invalid loss detected (NaN or Inf)! Skipping loss accumulation for batch {i}.")
                else:
                    total_loss += loss.item() * batch.num_graphs
                    num_samples += batch.num_graphs
            except Exception as e:
                logging.error(f"[Eval] Error during loss calculation: {e}. Skipping batch {i}.", exc_info=True)
                continue
            if median_q_idx != -1:
                try:
                    preds_median_scaled = out[:, median_q_idx].cpu().numpy()
                    targets_scaled = target.cpu().numpy()
                    current_batch_size = batch.num_graphs
                    num_nodes_in_batch_expected = batch.num_graphs * num_nodes
                    if batch.num_nodes != num_nodes_in_batch_expected:
                        logging.warning(f"Node count mismatch in batch {i}. Expected {num_nodes_in_batch_expected}, got {batch.num_nodes}. Skipping inverse transform.")
                        continue
                    preds_median_scaled_reshaped = preds_median_scaled.reshape(batch.num_graphs, num_nodes)
                    targets_scaled_reshaped = targets_scaled.reshape(batch.num_graphs, num_nodes)
                    preds_orig_batch = np.zeros_like(preds_median_scaled_reshaped)
                    targets_orig_batch = np.zeros_like(targets_scaled_reshaped)
                    for node_idx in range(num_nodes):
                        ticker = tickers[node_idx]
                        if ticker in scalers and target_feature in scalers[ticker]:
                            scaler_target = scalers[ticker][target_feature]
                            preds_node_scaled = preds_median_scaled_reshaped[:, node_idx].reshape(-1, 1)
                            targets_node_scaled = targets_scaled_reshaped[:, node_idx].reshape(-1, 1)
                            valid_preds = ~np.isnan(preds_node_scaled) & ~np.isinf(preds_node_scaled)
                            valid_targets = ~np.isnan(targets_node_scaled) & ~np.isinf(targets_node_scaled)
                            preds_orig_batch[:, node_idx] = np.nan
                            targets_orig_batch[:, node_idx] = np.nan
                            if np.any(valid_preds):
                                valid_scaled_preds_1d = preds_node_scaled[valid_preds]
                                valid_scaled_preds_2d = valid_scaled_preds_1d.reshape(-1, 1)
                                preds_orig_batch[valid_preds.flatten(), node_idx] = scaler_target.inverse_transform(valid_scaled_preds_2d).flatten()
                            if np.any(valid_targets):
                                valid_scaled_targets_1d = targets_node_scaled[valid_targets]
                                valid_scaled_targets_2d = valid_scaled_targets_1d.reshape(-1, 1)
                                targets_orig_batch[valid_targets.flatten(), node_idx] = scaler_target.inverse_transform(valid_scaled_targets_2d).flatten()
                        else:
                            preds_orig_batch[:, node_idx] = np.nan
                            targets_orig_batch[:, node_idx] = np.nan
                    all_preds_orig.append(preds_orig_batch)
                    all_targets_orig.append(targets_orig_batch)
                except Exception as e:
                    logging.error(f"Error during inverse scaling in batch {i}: {e}", exc_info=True)
                    nan_array = np.full((batch.num_graphs, num_nodes), np.nan)
                    all_preds_orig.append(nan_array)
                    all_targets_orig.append(nan_array)
    avg_loss = total_loss / num_samples if num_samples > 0 else float('inf')
    rmse, mae = float('nan'), float('nan')
    final_preds_orig_array = None
    final_targets_orig_array = None
    if all_preds_orig and all_targets_orig:
        try:
            final_preds_orig_array = np.concatenate(all_preds_orig, axis=0)
            final_targets_orig_array = np.concatenate(all_targets_orig, axis=0)
            valid_mask = ~np.isnan(final_preds_orig_array) & ~np.isnan(final_targets_orig_array)
            if np.sum(valid_mask) > 0:
                rmse = np.sqrt(np.mean((final_preds_orig_array[valid_mask] - final_targets_orig_array[valid_mask])**2))
                mae = np.mean(np.abs(final_preds_orig_array[valid_mask] - final_targets_orig_array[valid_mask]))
            else:
                logging.warning("No valid (non-NaN) points for metric calculation.")
        except Exception as e:
            logging.error(f"Error calculating final metrics: {e}", exc_info=True)
    return avg_loss, rmse, mae, final_preds_orig_array, final_targets_orig_array

def objective(trial: optuna.trial.Trial, train_loader: DataLoader, val_loader: Optional[DataLoader], scalers: Dict[str, Dict[str, StandardScaler]], tickers: List[str], num_features: int, edge_feature_dim: int, num_quantiles: List[float], target_feature: str, quantiles: List[float], median_q_idx: int, device: torch.device, max_epochs: int, patience: int) -> float:
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    node_hidden_dim = trial.suggest_categorical("node_hidden_dim", [32, 64, 128])
    gat_heads = 1
    logging.info(f"Trial {trial.number}: Forcing gat_heads = 1 for TRT compatibility test.")
    try:
        model = StockForecastGNN(num_features=num_features, node_hidden_dim=node_hidden_dim, gat_heads=gat_heads, edge_dim=edge_feature_dim, num_quantiles=num_quantiles).to(device)
    except ValueError as e:
        logging.warning(f"Trial {trial.number}: Skipping due to invalid model parameters: {e}")
        return float('inf')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = quantile_loss
    best_val_loss = float('inf')
    epochs_no_improve = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        total_train_loss = 0
        num_train_batches = 0
        pbar_train = tqdm(train_loader, desc=f"Trial {trial.number} Epoch {epoch}/{max_epochs} [Train]", leave=False, disable=True)
        for batch in pbar_train:
            batch = batch.to(device)
            optimizer.zero_grad()
            try:
                out = model(batch)
                loss = criterion(out, batch.y, quantiles)
                if torch.isnan(loss) or torch.isinf(loss):
                    logging.warning(f"Trial {trial.number} Epoch {epoch}: NaN/Inf train loss. Skipping backward.")
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_train_loss += loss.item()
                num_train_batches += 1
            except RuntimeError as e:
                logging.error(f"Trial {trial.number} Epoch {epoch}: RuntimeError during training: {e}", exc_info=True)
                if "CUDA out of memory" in str(e):
                    logging.error("CUDA OOM in Trial. Pruning.")
                    raise optuna.TrialPruned("CUDA OOM")
                continue
        avg_train_loss = total_train_loss / num_train_batches if num_train_batches > 0 else float('inf')
        avg_val_loss = float('inf')
        if val_loader:
            avg_val_loss, _, _, _, _ = evaluate(model, val_loader, criterion, device, scalers, tickers, target_feature, quantiles, median_q_idx)
            if np.isnan(avg_val_loss):
                avg_val_loss = float('inf')
            trial.report(avg_val_loss, epoch)
            if trial.should_prune():
                logging.debug(f"Trial {trial.number} pruned at epoch {epoch}.")
                raise optuna.TrialPruned()
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logging.debug(f"Trial {trial.number}: Early stopping triggered at epoch {epoch}.")
                break
        else:
            best_val_loss = avg_train_loss
            trial.report(avg_train_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        logging.debug(f"Trial {trial.number} Epoch {epoch}/{max_epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
    return best_val_loss

if __name__ == "__main__":
    logging.info(f"Using device: {DEVICE}")
    logging.info(f"Config: SEQ_LEN={SEQ_LEN}, PRED_HORIZON={PRED_HORIZON}, BATCH_SIZE={BATCH_SIZE}")
    logging.info(f"Optuna Trials: {N_TRIALS}, Max Epochs/Trial: {MAX_EPOCHS}, Patience: {EARLY_STOPPING_PATIENCE}")
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    base_run_name = f"{current_time}_QuantGNN_Heads1_MeanAggr"
    run_output_dir = BASE_OUTPUT_DIR / base_run_name
    run_output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Run outputs (Optuna DB, final logs/model) will be saved to: {run_output_dir}")
    hook_handle = None
    final_model = None
    writer = None
    study = None
    final_test_results = {'hparam/test_loss': -999.0, 'hparam/test_rmse': -999.0, 'hparam/test_mae': -999.0}
    last_final_epoch = 0
    tickers_list = []
    scalers = {}
    best_params = {}
    try:
        ticker_files = find_data_files(DATA_FOLDER, FILE_SUFFIX)
        if not ticker_files:
            raise FileNotFoundError(f"No data files found for suffix '{FILE_SUFFIX}' in '{DATA_FOLDER}'.")
        tickers_list = sorted(list(ticker_files.keys()))
        logging.info(f"Processing tickers: {', '.join(tickers_list)}")
        combined_data = load_and_align_data(ticker_files, FEATURES)
        if combined_data.empty:
            raise ValueError("Combined data is empty after loading/alignment/NaN handling.")
        scaled_data, scalers = scale_data(combined_data)
        num_features = len(FEATURES)
        edge_feature_dim = 1
        scaler_save_path = run_output_dir / f"scalers_{base_run_name}.pkl"
        try:
            with open(scaler_save_path, 'wb') as f:
                pickle.dump(scalers, f)
            logging.info(f"Scalers saved to '{scaler_save_path}'")
        except Exception as e:
            logging.error(f"Failed to save scalers: {e}", exc_info=True)
        train_size = int(len(scaled_data) * 0.8)
        val_size = int(len(scaled_data) * 0.1)
        train_data_df = scaled_data.iloc[:train_size]
        val_data_df = scaled_data.iloc[train_size : train_size + val_size]
        test_data_df = scaled_data.iloc[train_size + val_size :]
        logging.info(f"Data split: Train={len(train_data_df)}, Val={len(val_data_df)}, Test={len(test_data_df)}")
        min_len_needed = SEQ_LEN + PRED_HORIZON
        if len(train_data_df) < min_len_needed:
            raise ValueError(f"Insufficient training data ({len(train_data_df)} rows) after split.")
        if len(val_data_df) < min_len_needed:
            logging.warning(f"Validation split too small ({len(val_data_df)} rows). Optuna/Final training might lack reliable validation.")
            val_dataset = None
        else:
            val_dataset = StockDataset(val_data_df, tickers_list, FEATURES, TARGET_FEATURE, SEQ_LEN, PRED_HORIZON, CORR_THRESHOLD)
        if len(test_data_df) < min_len_needed:
            raise ValueError(f"Insufficient test data ({len(test_data_df)} rows) after split.")
        train_dataset = StockDataset(train_data_df, tickers_list, FEATURES, TARGET_FEATURE, SEQ_LEN, PRED_HORIZON, CORR_THRESHOLD)
        test_dataset = StockDataset(test_data_df, tickers_list, FEATURES, TARGET_FEATURE, SEQ_LEN, PRED_HORIZON, CORR_THRESHOLD)
        if len(train_dataset) == 0: raise ValueError("Training dataset empty.")
        if val_dataset is None or len(val_dataset) == 0:
            logging.warning("Validation dataset empty or too small. Optuna will use train loss (less effective). Final training won't use validation for checkpointing/early stopping.")
            val_dataset = None
        if len(test_dataset) == 0: raise ValueError("Test dataset empty.")
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True, pin_memory=True if DEVICE=='cuda' else False)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True if DEVICE=='cuda' else False) if val_dataset else None
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True if DEVICE=='cuda' else False)
        logging.info(f"--- Starting Optuna Study ({N_TRIALS} trials) ---")
        db_path = run_output_dir / f"optuna_study_{base_run_name}.db"
        study = optuna.create_study(study_name=base_run_name, direction="minimize", storage=f"sqlite:///{db_path}", load_if_exists=True, pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3, interval_steps=1))
        objective_with_args = lambda trial: objective(trial, train_loader, val_loader, scalers, tickers_list, num_features, edge_feature_dim, NUM_QUANTILES, TARGET_FEATURE, QUANTILES, MEDIAN_Q_IDX, DEVICE, MAX_EPOCHS, EARLY_STOPPING_PATIENCE)
        study.optimize(objective_with_args, n_trials=N_TRIALS, timeout=OPTUNA_TIMEOUT, show_progress_bar=True)
        logging.info("--- Optuna Study Finished ---")
        logging.info(f"Number of finished trials: {len(study.trials)}")
        if study.best_trial:
            logging.info(f"Best trial number: {study.best_trial.number}")
            logging.info("Best hyperparameters found:")
            best_params = study.best_params
            if 'gat_heads' not in best_params:
                fixed_gat_heads = 1
                logging.info(f"Manually adding fixed gat_heads={fixed_gat_heads} to best_params dict.")
                best_params['gat_heads'] = fixed_gat_heads
            for key, value in best_params.items():
                logging.info(f"  {key}: {value}")
            logging.info(f"Best validation loss: {study.best_value:.6f}")
        else:
            logging.error("Optuna study finished without any successful trials. Cannot determine best parameters.")
            best_params = {}
        logging.info("--- Starting Final Training with Best Hyperparameters ---")
        if not best_params:
            raise RuntimeError("Optuna study did not yield any valid parameters. Cannot proceed with final training.")
        final_run_output_dir = run_output_dir / "final_run"
        final_run_output_dir.mkdir(exist_ok=True)
        writer = SummaryWriter(log_dir=str(final_run_output_dir))
        logging.info(f"Final run TensorBoard logs saving to: {final_run_output_dir}")
        final_model = StockForecastGNN(num_features=num_features, node_hidden_dim=best_params['node_hidden_dim'], gat_heads=best_params.get('gat_heads', 1), edge_dim=edge_feature_dim, num_quantiles=NUM_QUANTILES).to(DEVICE)
        final_optimizer = torch.optim.Adam(final_model.parameters(), lr=best_params['lr'])
        criterion = quantile_loss
        logging.info("--- Final Model Architecture (aggr='mean', heads=1) ---")
        print(final_model)
        num_params = sum(p.numel() for p in final_model.parameters() if p.requires_grad)
        logging.info(f"Total Trainable Parameters: {num_params:,}")
        best_final_val_loss = float('inf')
        final_epochs_no_improve = 0
        final_model_save_path = final_run_output_dir / f"best_final_model_{base_run_name}.pth"
        for epoch in range(1, MAX_EPOCHS + 1):
            last_final_epoch = epoch
            final_model.train()
            total_train_loss = 0
            num_train_batches = 0
            pbar = tqdm(train_loader, desc=f"Final Epoch {epoch}/{MAX_EPOCHS} [Train]", leave=False)
            for batch in pbar:
                batch = batch.to(DEVICE)
                final_optimizer.zero_grad()
                try:
                    out = final_model(batch)
                    loss = criterion(out, batch.y, QUANTILES)
                    if torch.isnan(loss) or torch.isinf(loss):
                        logging.warning(f"Final Epoch {epoch}: NaN/Inf train loss. Skipping backward.")
                        continue
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(final_model.parameters(), max_norm=1.0)
                    final_optimizer.step()
                    total_train_loss += loss.item()
                    num_train_batches += 1
                    pbar.set_postfix(loss=f"{loss.item():.4f}")
                except RuntimeError as e:
                    logging.error(f"Final Epoch {epoch}: RuntimeError during training: {e}", exc_info=True)
                    if "CUDA out of memory" in str(e): torch.cuda.empty_cache()
                    continue
            avg_train_loss = total_train_loss / num_train_batches if num_train_batches > 0 else float('inf')
            logging.info(f"Final Epoch {epoch}/{MAX_EPOCHS} - Avg Training Loss: {avg_train_loss:.6f}")
            if writer: writer.add_scalar('Loss/Train_Final', avg_train_loss, epoch)
            if epoch % HISTOGRAM_LOG_FREQ == 0 and writer:
                for name, param in final_model.named_parameters():
                    if param.requires_grad and param.data is not None:
                        if not(torch.isnan(param.data).any() or torch.isinf(param.data).any()):
                            writer.add_histogram(f'Parameters_Final/{name}', param.data.cpu(), epoch)
                        else:
                            logging.warning(f"NaN/Inf in param '{name}' final epoch {epoch}. Skipping histogram.")
            avg_val_loss, val_rmse, val_mae = float('inf'), float('nan'), float('nan')
            if val_loader:
                avg_val_loss, val_rmse, val_mae, _, _ = evaluate(final_model, val_loader, criterion, DEVICE, scalers, tickers_list, TARGET_FEATURE, QUANTILES, MEDIAN_Q_IDX)
                logging.info(f"Final Epoch {epoch}/{MAX_EPOCHS} - Avg Validation Loss: {avg_val_loss:.6f}, Val RMSE: {val_rmse:.4f}, Val MAE: {val_mae:.4f}")
                if writer:
                    if not np.isnan(avg_val_loss): writer.add_scalar('Loss/Validation_Final', avg_val_loss, epoch)
                    if not np.isnan(val_rmse): writer.add_scalar('RMSE/Validation_MedianQ_Final', val_rmse, epoch)
                    if not np.isnan(val_mae): writer.add_scalar('MAE/Validation_MedianQ_Final', val_mae, epoch)
                if not np.isnan(avg_val_loss) and avg_val_loss < best_final_val_loss:
                    best_final_val_loss = avg_val_loss
                    final_epochs_no_improve = 0
                    try:
                        torch.save(final_model.state_dict(), final_model_save_path)
                        logging.info(f"New best final validation loss: {best_final_val_loss:.6f}. Model saved to '{final_model_save_path}'")
                    except Exception as e:
                        logging.error(f"Error saving best final model: {e}", exc_info=True)
                elif not np.isnan(avg_val_loss):
                    final_epochs_no_improve += 1
                    logging.info(f"Final Val loss did not improve for {final_epochs_no_improve} epoch(s). Best: {best_final_val_loss:.6f}")
                if final_epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                    logging.info(f"Final training early stopping triggered after {epoch} epochs.")
                    break
            else:
                if epoch == MAX_EPOCHS:
                    try:
                        torch.save(final_model.state_dict(), final_model_save_path.with_name(f"last_final_model_{base_run_name}.pth"))
                        logging.info(f"Final training finished. Saved last model.")
                    except Exception as e:
                        logging.error(f"Error saving last final model: {e}", exc_info=True)
        logging.info("--- Final Training Finished ---")
        best_model_path_final = final_run_output_dir / f"best_final_model_{base_run_name}.pth"
        last_model_path_final = final_run_output_dir / f"last_final_model_{base_run_name}.pth"
        model_path_to_load = None
        if val_loader and best_model_path_final.exists():
            model_path_to_load = best_model_path_final
            logging.info(f"Loading best final model based on validation: '{model_path_to_load}'")
        elif not val_loader and last_model_path_final.exists():
            model_path_to_load = last_model_path_final
            logging.info(f"Loading last final model (no validation): '{model_path_to_load}'")
        elif best_model_path_final.exists():
            model_path_to_load = best_model_path_final
        else:
            logging.warning(f"No saved final model file found. Evaluating with current model state.")
        if model_path_to_load and final_model:
            try:
                final_model.load_state_dict(torch.load(model_path_to_load, map_location=DEVICE))
                logging.info(f"Loaded final model '{model_path_to_load}' for test evaluation.")
            except Exception as e:
                logging.error(f"Error loading final model state dict: {e}. Evaluating with current state.", exc_info=True)
        elif not final_model:
            raise RuntimeError("Final model not initialized successfully.")
        if hasattr(final_model, 'gat') and isinstance(final_model.gat, StockGAT):
            try:
                hook_handle = final_model.gat.gat.register_forward_hook(gnn_hook_fn)
                logging.info(f"Registered forward hook on final model GAT layer: {final_model.gat.gat}")
            except Exception as e:
                logging.error(f"Failed to register forward hook on final model GAT: {e}", exc_info=True)
                hook_handle = None
        else:
            logging.warning("Could not find final_model.gat.gat layer to register hook.")
        logging.info("--- Starting Final Test Set Evaluation ---")
        final_model.eval()
        total_test_loss = 0
        num_test_samples = 0
        all_test_preds_orig = []
        all_test_targets_orig = []
        final_gnn_embeddings = None
        with torch.no_grad():
            for i, batch in enumerate(tqdm(test_loader, desc="Final Testing", leave=False)):
                batch = batch.to(DEVICE)
                if batch.num_graphs == 0: continue
                if i == 0 and hook_handle is not None:
                    captured_gnn_output = None
                    try:
                        out = final_model(batch)
                        if captured_gnn_output is not None:
                            final_gnn_embeddings = captured_gnn_output
                            logging.info(f"Hook captured GNN output shape: {final_gnn_embeddings.shape}")
                        else: logging.warning("Hook ran but captured_gnn_output is None.")
                    except Exception as e:
                        logging.error(f"[Test Hook] Error during model forward pass: {e}", exc_info=True)
                        try: out = final_model(batch)
                        except Exception as e2:
                            logging.error(f"[Test Hook Retry] Error: {e2}. Skipping batch.", exc_info=True)
                            continue
                else:
                    try: out = final_model(batch)
                    except Exception as e:
                        logging.error(f"[Test] Error during model forward pass: {e}. Skipping batch.", exc_info=True)
                        continue
                target = batch.y
                try:
                    loss = criterion(out, target, QUANTILES)
                    if not (torch.isnan(loss) or torch.isinf(loss)):
                        total_test_loss += loss.item() * batch.num_graphs
                        num_test_samples += batch.num_graphs
                    else: logging.warning("[Test] Invalid loss detected.")
                except Exception as e:
                    logging.error(f"[Test] Error during loss calc: {e}. Skipping batch.", exc_info=True)
                    continue
                if MEDIAN_Q_IDX != -1:
                    try:
                        preds_median_scaled = out[:, MEDIAN_Q_IDX].cpu().numpy()
                        targets_scaled = target.cpu().numpy()
                        current_batch_size = batch.num_graphs
                        num_nodes = len(tickers_list)
                        num_nodes_in_batch_expected = batch.num_graphs * num_nodes
                        if batch.num_nodes != num_nodes_in_batch_expected:
                            logging.warning(f"Node count mismatch in batch {i}. Expected {num_nodes_in_batch_expected}, got {batch.num_nodes}. Skipping inverse transform.")
                            continue
                        preds_med_reshaped = preds_median_scaled.reshape(current_batch_size, num_nodes)
                        targets_reshaped = targets_scaled.reshape(current_batch_size, num_nodes)
                        preds_orig_batch = np.zeros_like(preds_med_reshaped)
                        targets_orig_batch = np.zeros_like(targets_reshaped)
                        for node_idx in range(num_nodes):
                            ticker = tickers_list[node_idx]
                            if ticker in scalers and TARGET_FEATURE in scalers[ticker]:
                                scaler_target = scalers[ticker][TARGET_FEATURE]
                                preds_node_scaled = preds_med_reshaped[:, node_idx].reshape(-1, 1)
                                targets_node_scaled = targets_reshaped[:, node_idx].reshape(-1, 1)
                                valid_preds = ~np.isnan(preds_node_scaled) & ~np.isinf(preds_node_scaled)
                                valid_targets = ~np.isnan(targets_node_scaled) & ~np.isinf(targets_node_scaled)
                                preds_orig_batch[:, node_idx] = np.nan
                                targets_orig_batch[:, node_idx] = np.nan
                                if np.any(valid_preds):
                                    valid_scaled_preds_1d = preds_node_scaled[valid_preds]
                                    valid_scaled_preds_2d = valid_scaled_preds_1d.reshape(-1, 1)
                                    preds_orig_batch[valid_preds.flatten(), node_idx] = scaler_target.inverse_transform(valid_scaled_preds_2d).flatten()
                                if np.any(valid_targets):
                                    valid_scaled_targets_1d = targets_node_scaled[valid_targets]
                                    valid_scaled_targets_2d = valid_scaled_targets_1d.reshape(-1, 1)
                                    targets_orig_batch[valid_targets.flatten(), node_idx] = scaler_target.inverse_transform(valid_scaled_targets_2d).flatten()
                            else:
                                preds_orig_batch[:, node_idx] = np.nan
                                targets_orig_batch[:, node_idx] = np.nan
                        all_test_preds_orig.append(preds_orig_batch)
                        all_test_targets_orig.append(targets_orig_batch)
                    except Exception as e:
                        logging.error(f"Error during test inverse scaling: {e}", exc_info=True)
                        all_test_preds_orig.append(np.full((batch.num_graphs, len(tickers_list)), np.nan))
                        all_test_targets_orig.append(np.full((batch.num_graphs, len(tickers_list)), np.nan))
        test_loss = total_test_loss / num_test_samples if num_test_samples > 0 else float('nan')
        test_rmse, test_mae = float('nan'), float('nan')
        test_preds_orig, test_targets_orig = None, None
        if all_test_preds_orig and all_test_targets_orig:
            try:
                test_preds_orig = np.concatenate(all_test_preds_orig, axis=0)
                test_targets_orig = np.concatenate(all_test_targets_orig, axis=0)
                valid_mask = ~np.isnan(test_preds_orig) & ~np.isnan(test_targets_orig)
                if np.sum(valid_mask) > 0:
                    test_rmse = np.sqrt(np.mean((test_preds_orig[valid_mask] - test_targets_orig[valid_mask])**2))
                    test_mae = np.mean(np.abs(test_preds_orig[valid_mask] - test_targets_orig[valid_mask]))
            except Exception as e: logging.error(f"Error calculating final test metrics: {e}", exc_info=True)
        logging.info(f'Final Test Loss (Quantile): {test_loss:.6f}')
        logging.info(f'Final Test RMSE (Original Scale, Median Q): {test_rmse:.4f}')
        logging.info(f'Final Test MAE (Original Scale, Median Q): {test_mae:.4f}')
        final_test_results = {'hparam/test_loss': test_loss if not np.isnan(test_loss) else -999.0, 'hparam/test_rmse': test_rmse if not np.isnan(test_rmse) else -999.0, 'hparam/test_mae': test_mae if not np.isnan(test_mae) else -999.0}
        if writer:
            if final_gnn_embeddings is not None:
                logging.info(f"Logging GNN embeddings to TensorBoard Projector (shape: {final_gnn_embeddings.shape})")
                num_embeddings = final_gnn_embeddings.shape[0]
                num_nodes_in_batch = len(tickers_list)
                if num_embeddings > 0 and num_nodes_in_batch > 0 and num_embeddings % num_nodes_in_batch == 0:
                    num_samples_in_batch = num_embeddings // num_nodes_in_batch
                    metadata = [f"S{s}_N{n}_{tickers_list[n]}" for s in range(num_samples_in_batch) for n in range(num_nodes_in_batch)]
                    if len(metadata) == num_embeddings:
                        try:
                            writer.add_embedding(final_gnn_embeddings, metadata=metadata, global_step=last_final_epoch, tag="GAT_Output_Embeddings_Test")
                            logging.info("Logged embeddings.")
                        except Exception as e: logging.error(f"Error logging embeddings: {e}", exc_info=True)
                    else: logging.error(f"Metadata label count mismatch for embeddings. Expected {num_embeddings}, got {len(metadata)}")
                else: logging.warning(f"Embeddings count issue (Embeddings: {num_embeddings}, Nodes: {num_nodes_in_batch}). Skipping logging.")
            else: logging.warning("No GNN embeddings captured for logging.")
            if test_preds_orig is not None and test_targets_orig is not None and MEDIAN_Q_IDX != -1:
                logging.info("Generating prediction plots for TensorBoard...")
                num_test_samples_plot, num_nodes_plot = test_preds_orig.shape
                if num_nodes_plot == len(tickers_list):
                    try:
                        max_plots = 20
                        plot_indices = range(min(num_nodes_plot, max_plots))
                        for i in plot_indices:
                            ticker = tickers_list[i]
                            fig, ax = plt.subplots(figsize=(12, 6))
                            preds_ticker = test_preds_orig[:, i]
                            targets_ticker = test_targets_orig[:, i]
                            valid_idx = ~np.isnan(preds_ticker) & ~np.isnan(targets_ticker)
                            time_axis = np.arange(num_test_samples_plot)
                            if np.sum(valid_idx) > 0:
                                ax.plot(time_axis[valid_idx], targets_ticker[valid_idx], label='Actual', color='blue', alpha=0.7, marker='.', markersize=3, linestyle='')
                                ax.plot(time_axis[valid_idx], preds_ticker[valid_idx], label=f'Predicted (Q{QUANTILES[MEDIAN_Q_IDX]})', color='red', alpha=0.7, marker='x', markersize=3, linestyle='')
                                ax.set_title(f'Test Set: {ticker} - Actual vs Predicted')
                                ax.set_xlabel('Test Sequence Index')
                                ax.set_ylabel(f'{TARGET_FEATURE} (Original Scale)')
                                ax.legend(); ax.grid(True)
                                writer.add_figure(f'Predictions_vs_Actuals/Ticker_{ticker}', fig, global_step=last_final_epoch)
                            plt.close(fig)
                        logging.info(f"Added prediction plots for {len(plot_indices)} tickers.")
                    except Exception as e: logging.error(f"Error generating plots: {e}", exc_info=True)
                else: logging.warning("Plotting skipped: Shape mismatch.")
            else: logging.warning("Plotting skipped: Data unavailable.")
    except optuna.exceptions.TrialPruned as e:
        logging.info(f"Optuna trial pruned: {e}")
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        logging.error(f"Execution failed: {e}", exc_info=True)
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)
    finally:
        if hook_handle:
            try: hook_handle.remove(); logging.info("Removed forward hook.")
            except Exception as e: logging.error(f"Error removing hook: {e}", exc_info=True)
        if writer and study and best_params:
            hparam_dict = {'seq_len': SEQ_LEN, 'pred_horizon': PRED_HORIZON, 'corr_threshold': CORR_THRESHOLD, 'batch_size': BATCH_SIZE, 'epochs_run_final': last_final_epoch, 'num_quantiles': NUM_QUANTILES, 'num_tickers': len(tickers_list), 'optuna_trials': N_TRIALS, 'optuna_best_lr': best_params.get('lr', -1), 'optuna_best_node_hidden_dim': best_params.get('node_hidden_dim', -1), 'optuna_best_gat_heads': best_params.get('gat_heads', -1), 'gat_aggregation': 'mean'}
            hparam_dict_numeric = {k: v for k, v in hparam_dict.items() if isinstance(v, (int, float))}
            metrics_dict = {k: v if isinstance(v, (int, float)) and np.isfinite(v) else -999.0 for k, v in final_test_results.items()}
            try:
                writer.add_hparams(hparam_dict_numeric, metrics_dict, run_name='.')
                logging.info("Logged hyperparameters and final metrics to TensorBoard.")
            except Exception as e:
                logging.warning(f"Could not log hyperparameters to TensorBoard: {e}", exc_info=True)
        if best_params and 'final_run_output_dir' in locals() and final_run_output_dir:
            params_save_path = final_run_output_dir / f"best_params_{base_run_name}.pkl"
            try:
                with open(params_save_path, 'wb') as f:
                    pickle.dump(best_params, f)
                logging.info(f"Best hyperparameters saved to '{params_save_path}'")
            except NameError:
                logging.warning("Could not save best_params: base_run_name not defined in this scope.")
            except Exception as e:
                logging.error(f"Failed to save best hyperparameters: {e}", exc_info=True)
        elif not best_params:
            logging.warning("Optuna did not find best parameters (study might have failed or params were empty). Cannot save params.")
        if writer:
            writer.close()
            logging.info("TensorBoard writer closed.")
        logging.info("--- Script Finished ---")