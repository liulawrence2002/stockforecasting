# --- START OF FILE GNN_forecast_tuned.py ---
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data, Dataset, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import dense_to_sparse
from torch.utils.tensorboard import SummaryWriter

import pandas as pd
import numpy as np
import os
import glob
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging
import random
import datetime
import matplotlib.pyplot as plt
import inspect # Needed for checking GATv2Conv signature
import optuna # <<< ADDED: Optuna for hyperparameter tuning
# from optuna.integration import PyTorchLightningPruningCallback # Example Pruning (if using Lightning later) - Using manual pruning here # Not used currently
from optuna.exceptions import TrialPruned # <<< ADDED: For pruning
from tqdm import tqdm # <<< ADDED: For progress bar
import time # <<< ADDED: For timing trials

# --- Basic Configuration & Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler()) # Pipe Optuna logs to std out
optuna.logging.get_logger("optuna").setLevel(logging.INFO)

# --- Configuration ---
# Data Parameters
DATA_FOLDER = "GNN_data"
TICKERS = None # ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'AMD'] # Set to None to load all stocks
SEQUENCE_LENGTH = 30 # Fixed for simplicity during tuning, could be tuned but requires re-sequencing
PREDICTION_HORIZON = 3 # Fixed
FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume']
TARGET = 'Close'

# Graph Construction Parameters (Now potentially tunable)
# CORRELATION_THRESHOLD = 0.2 # Moved to objective
# CORRELATION_WINDOW = 70   # Moved to objective

# --- Model Hyperparameters (Define search space in objective) ---
# LSTM Parameters
# LSTM_HIDDEN_DIM = 128
# LSTM_LAYERS = 2
# GNN Parameters
# GNN_HIDDEN_DIM = 64
# GNN_LAYERS = 2
# GNN_HEADS = 4
# Fusion Parameters
# FUSION_HIDDEN_DIM = 96
# General Parameters
OUTPUT_DIM = 1
# DROPOUT_RATE = 0.3

# Training Parameters (Some tunable)
# LEARNING_RATE = 0.001
EPOCHS = 75 # Epochs *per trial*
# BATCH_SIZE = 32
TRAIN_SPLIT = 0.7
VALIDATION_SPLIT = 0.15
SEED = 1
# LOSS_ALPHA = 0.05 # Tunable

# --- Optuna Configuration ---
N_TRIALS = 10 # Total number of tuning trials to aim for
OPTUNA_TIMEOUT = 3600 * 4 # Timeout for the study (e.g., 2 hours)
METRIC_TO_OPTIMIZE = "val_rmse" # or "val_loss"
PRUNING_ENABLED = True # Enable/disable trial pruning

# --- TensorBoard Configuration ---
TENSORBOARD_LOG_DIR_BASE = "runs/optuna_gnn_lstm_stock_forecast" # Base directory for runs
# HISTOGRAM_LOG_FREQ = 10 # Keep logging histograms, but maybe less frequently during tuning

# --- Set Seed for Reproducibility (Initial Setup) ---
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # Note: Setting deterministic = True can slow down training.
    # For tuning, benchmark = True and deterministic = False is often preferred.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


# --- Helper Functions ---
# load_and_align_data, create_sequences, build_correlation_graph remain mostly the same
# (build_correlation_graph now takes threshold/window as args)
def load_and_align_data(data_folder: str, tickers: list = None) -> (pd.DataFrame, list):
    """Loads all CSVs, aligns timestamps, returns combined DataFrame and ticker order."""
    all_files = glob.glob(os.path.join(data_folder, "*.csv"))
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in folder: {data_folder}")

    if tickers:
        tickers_upper = [t.upper() for t in tickers] # Ensure comparison is case-insensitive
        filtered_files = []
        tickers_found = set()
        for ticker in tickers_upper:
            found = False
            # Allow flexibility in filename (e.g., AAPL_30minute.csv, AAPL_1day.csv)
            pattern = os.path.join(data_folder, f"{ticker}_*.csv")
            ticker_files = glob.glob(pattern)
            if ticker_files:
                 # Prefer shorter filenames if multiple match (e.g., AAPL_1day.csv over AAPL_1day_extra.csv)
                 ticker_files.sort(key=len)
                 filtered_files.append(ticker_files[0]) # Add the best match
                 tickers_found.add(ticker)
                 found = True

            if not found:
                logging.warning(f"No CSV file found matching pattern for ticker: {ticker} in {data_folder}")

        # Check if all requested tickers were found
        missing_requested = set(tickers_upper) - tickers_found
        if missing_requested:
             logging.warning(f"Could not find data files for requested tickers: {missing_requested}")

        all_files = filtered_files
        if not all_files:
             raise FileNotFoundError(f"No CSV files found for the specified tickers: {tickers}")
        logging.info(f"Found {len(all_files)} files for specified tickers.")
    else:
        logging.info(f"TICKERS is None. Loading all CSV files found in {data_folder}.")


    data_frames = {}
    ticker_order = []
    common_index = None

    logging.info(f"Loading data for tickers: {[os.path.basename(f).split('_')[0] for f in all_files]}")

    for f_path in sorted(all_files): # Sort for consistent ticker order
        ticker = os.path.basename(f_path).split('_')[0]
        try:
            df = pd.read_csv(f_path, index_col='Timestamp_ms', parse_dates=False)
            df.index = pd.to_datetime(df.index, unit='ms', utc=True)

            # Check for required features
            missing_features = [f for f in FEATURES if f not in df.columns]
            if missing_features:
                 logging.warning(f"Missing features {missing_features} in {f_path}. Skipping ticker {ticker}.")
                 continue

            df = df[FEATURES]
            df.columns = [f"{ticker}_{col}" for col in df.columns]
            df = df.dropna()

            if not df.empty:
                data_frames[ticker] = df
                ticker_order.append(ticker)
                if common_index is None:
                    common_index = df.index
                else:
                    common_index = common_index.intersection(df.index)
            else:
                logging.warning(f"DataFrame for {ticker} is empty after loading/dropping NA.")

        except Exception as e:
            logging.error(f"Error loading or processing file {f_path}: {e}")

    if not data_frames:
        raise ValueError("No valid data loaded. Check CSV files and specified tickers.")
    if common_index is None or len(common_index) == 0:
         raise ValueError("No common timestamps found across the loaded data files. Check data alignment.")


    logging.info(f"Found {len(common_index)} common timestamps across {len(ticker_order)} potential tickers.")

    aligned_df_list = []
    final_ticker_order = [] # Build the final list based on successful alignment
    for ticker in ticker_order:
        if ticker in data_frames:
            # Reindex to common_index, dropping rows that don't exist in this df
            df_aligned = data_frames[ticker].reindex(common_index)
            # Crucially, check if *after* aligning to common index, data still exists
            if not df_aligned.isnull().all().all(): # Check if not all values are NaN
                 aligned_df_list.append(df_aligned)
                 final_ticker_order.append(ticker) # Add to final list only if data exists
            else:
                 logging.warning(f"Ticker {ticker} had no valid data at common timestamps after reindexing. Skipping.")
        else:
             # This case shouldn't happen if ticker_order is built correctly, but handle defensively
             logging.warning(f"Ticker {ticker} was in initial order but not in data_frames dict. Skipping.")


    if not aligned_df_list:
         raise ValueError("No dataframes left after alignment. Check common timestamps and individual file contents.")

    # Concatenate should now work reliably
    combined_df = pd.concat(aligned_df_list, axis=1)
    # It's possible some NaNs were introduced by reindexing if a ticker missed a few common timestamps
    # Decide on fill strategy: ffill, bfill, or keep NaNs and handle later? Let's ffill for simplicity here.
    combined_df = combined_df.ffill().bfill() # Forward fill then back fill remaining NaNs
    combined_df = combined_df.dropna() # Drop any rows that might still have NaNs (e.g., at the very beginning)

    if combined_df.empty:
         raise ValueError("Combined dataframe is empty after alignment and fillna. Check input data.")

    logging.info(f"Combined DataFrame shape after alignment and fill: {combined_df.shape} for {len(final_ticker_order)} tickers: {final_ticker_order}")
    return combined_df, final_ticker_order


def create_sequences(data: np.ndarray, sequence_length: int, prediction_horizon: int) -> (np.ndarray, np.ndarray):
    """Creates input sequences and corresponding targets."""
    xs, ys = [], []
    if data is None or len(data) <= sequence_length + prediction_horizon -1: # Added None check
        logging.warning(f"Data length ({len(data) if data is not None else 'None'}) too short for sequence_length ({sequence_length}) and prediction_horizon ({prediction_horizon}). Returning empty arrays.")
        return np.array(xs), np.array(ys)

    for i in range(len(data) - sequence_length - prediction_horizon + 1):
        x = data[i:(i + sequence_length)]
        y = data[i + sequence_length + prediction_horizon - 1] # Target is the value at the end of horizon
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Modified to accept window and threshold as arguments
def build_correlation_graph(data: pd.DataFrame, ticker_order: list, window: int, threshold: float, target_feature: str = 'Close') -> (torch.Tensor, torch.Tensor):
    """Builds graph edges based on static price correlation over a window."""
    num_nodes = len(ticker_order)
    close_cols = [f"{ticker}_{target_feature}" for ticker in ticker_order]

    missing_cols = [col for col in close_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Target columns missing in DataFrame for correlation: {missing_cols}. Tickers: {ticker_order}")

    close_prices = data[close_cols]

    actual_window = min(window, len(close_prices))
    logging.debug(f"Calculating static correlation matrix over last {actual_window} samples with threshold {threshold}...") # Debug level for trials
    if actual_window < 2:
        logging.error(f"Not enough data points ({actual_window} < 2) to calculate correlation. Returning empty graph.")
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.float)

    corr_matrix = close_prices.iloc[-actual_window:].corr()
    corr_matrix = corr_matrix.fillna(0)

    adj_matrix = np.abs(corr_matrix.values) >= threshold
    np.fill_diagonal(adj_matrix, False)

    edge_index_tensor = torch.tensor(adj_matrix, dtype=torch.bool)
    edge_index = dense_to_sparse(edge_index_tensor)[0]

    edge_weights = torch.empty((0,), dtype=torch.float)
    if edge_index.numel() > 0:
        # Ensure indices are within bounds before accessing corr_matrix
        valid_edges_mask = (edge_index[0] < corr_matrix.shape[0]) & (edge_index[1] < corr_matrix.shape[1])
        valid_edge_index = edge_index[:, valid_edges_mask]

        if valid_edge_index.shape[1] < edge_index.shape[1]:
            logging.warning(f"Filtered out {edge_index.shape[1] - valid_edge_index.shape[1]} edges with out-of-bounds indices during graph build.")
            edge_index = valid_edge_index

        # Access corr_matrix only with valid indices
        if edge_index.numel() > 0: # Check again after filtering
            edge_weights = corr_matrix.values[edge_index[0], edge_index[1]]
            edge_weights = torch.tensor(edge_weights, dtype=torch.float)

    logging.debug(f"Graph constructed with {edge_index.shape[1]} edges.") # Debug level for trials
    return edge_index, edge_weights

# --- PyTorch Geometric Dataset ---
class StockDataset(Dataset):
    def __init__(self, X, y, edge_index, edge_weight=None):
        super().__init__()
        # X Shape: (num_samples, num_nodes, sequence_length, num_features)
        # y Shape: (num_samples, num_nodes) Target value for each node
        self.X = X
        self.y = y
        self.edge_index = edge_index
        self.edge_weight = edge_weight

        # Add checks for None before accessing shape
        if self.X is None or self.y is None:
             raise ValueError("Input X or y to StockDataset cannot be None")
        if self.X.shape[0] != self.y.shape[0]:
             raise ValueError(f"Samples mismatch: X={self.X.shape[0]}, y={self.y.shape[0]}")
        if self.X.shape[1] != self.y.shape[1]:
             raise ValueError(f"Nodes mismatch: X={self.X.shape[1]}, y={self.y.shape[1]}")

    def len(self):
        return len(self.X)

    def get(self, idx):
        x_seq_sample = torch.tensor(self.X[idx], dtype=torch.float) # (num_nodes, seq_len, num_features)
        y_sample = torch.tensor(self.y[idx], dtype=torch.float).unsqueeze(-1) # (num_nodes, 1) target feature

        num_nodes_in_sample = x_seq_sample.shape[0]

        # Return a PyG Data object
        data = Data(x_seq=x_seq_sample, # Node features over sequence length
                    edge_index=self.edge_index,
                    edge_attr=self.edge_weight,
                    y=y_sample, # Target for each node
                    num_nodes=num_nodes_in_sample # Explicitly store num_nodes if needed later
                   )
        return data

# --- LSTM-GNN Fusion Model Definition ---
class StockLSTM_GNN(nn.Module):
    def __init__(self, num_features, lstm_hidden_dim, lstm_layers,
                 gnn_hidden_dim, gnn_layers, gnn_heads, fusion_hidden_dim,
                 output_dim, dropout_rate):
        super().__init__()
        self.lstm_hidden_dim = lstm_hidden_dim
        self.gnn_hidden_dim = gnn_hidden_dim
        self.gnn_heads = gnn_heads
        self.dropout_rate = dropout_rate

        # --- LSTM Path ---
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True, # Crucial: Input shape (batch, seq, feature)
            dropout=dropout_rate if lstm_layers > 1 else 0 # Add dropout between LSTM layers if more than 1
        )

        # --- GNN Path ---
        self.gnn_layers = nn.ModuleList()
        # Input to GNN is the output of LSTM
        gnn_input_dim = lstm_hidden_dim
        current_gnn_dim = gnn_input_dim

        # Determine if edge_dim should be used
        use_edge_dim = False
        try:
             sig = inspect.signature(GATv2Conv.__init__)
             if 'edge_dim' in sig.parameters: use_edge_dim = True
        except Exception: pass
        gat_edge_dim = 1 if use_edge_dim else None # Set edge_dim=1 if GATv2Conv supports it

        # GNN Input Layer
        # Ensure GATv2Conv can be created with potentially 0 input dim if lstm_hidden_dim is small? No, LSTM output is always > 0.
        # Ensure gnn_hidden_dim * gnn_heads is valid.
        if gnn_layers > 0: # Only add GNN layers if gnn_layers > 0
            self.gnn_layers.append(
                GATv2Conv(current_gnn_dim, gnn_hidden_dim, heads=gnn_heads, dropout=dropout_rate, add_self_loops=True, edge_dim=gat_edge_dim, concat=True) # Explicitly concat=True for multi-head output
            )
            current_gnn_dim = gnn_hidden_dim * gnn_heads # Output dim after concat

            # GNN Hidden Layers
            for i in range(gnn_layers - 1):
                # Check if dimensions are valid before creating layer
                if current_gnn_dim <= 0 or gnn_hidden_dim <= 0 or gnn_heads <= 0:
                     raise ValueError(f"Invalid GNN dimensions at layer {i+1}: In={current_gnn_dim}, Out={gnn_hidden_dim}, Heads={gnn_heads}")
                layer = GATv2Conv(current_gnn_dim, gnn_hidden_dim, heads=gnn_heads, dropout=dropout_rate, add_self_loops=True, edge_dim=gat_edge_dim, concat=True)
                self.gnn_layers.append(layer)
                current_gnn_dim = gnn_hidden_dim * gnn_heads
        else:
             # If gnn_layers is 0, the GNN output dimension is effectively 0 for concatenation later
             current_gnn_dim = 0


        # --- Fusion Path ---
        # Adjust fusion input based on whether GNN layers exist
        fusion_input_dim = lstm_hidden_dim + current_gnn_dim
        if fusion_input_dim <= 0: # Should not happen if LSTM dim > 0
             raise ValueError(f"Fusion input dimension is zero or negative: {fusion_input_dim}")
        if fusion_hidden_dim <= 0:
            raise ValueError(f"Invalid Fusion hidden dimension: {fusion_hidden_dim}")

        self.fusion_fc1 = nn.Linear(fusion_input_dim, fusion_hidden_dim)
        self.fusion_fc2 = nn.Linear(fusion_hidden_dim, output_dim)
        self.fusion_dropout = nn.Dropout(dropout_rate)


    def forward(self, data):
        # Input is now a PyG Batch object
        x_seq, edge_index, edge_attr, batch = data.x_seq, data.edge_index, data.edge_attr, data.batch

        # --- LSTM Path ---
        # Add check for empty input sequence
        if x_seq.shape[0] == 0:
             logging.warning("LSTM received empty input sequence.")
             num_nodes_in_batch = batch.max().item() + 1 if batch is not None and batch.numel() > 0 else 0
             return torch.zeros((num_nodes_in_batch, self.fusion_fc2.out_features), device=x_seq.device)


        lstm_out, (hn, cn) = self.lstm(x_seq)
        lstm_out_last = lstm_out[:, -1, :] # Shape: (batch_size * num_nodes, lstm_hidden_dim)


        # --- GNN Path ---
        if len(self.gnn_layers) > 0:
            gnn_x = lstm_out_last # Shape: (batch_size * num_nodes, lstm_hidden_dim)

            # Prepare edge attributes if needed by GATv2Conv
            current_edge_attr = edge_attr
            first_gnn_layer = self.gnn_layers[0]
            expects_edge_dim = hasattr(first_gnn_layer, 'edge_dim') and first_gnn_layer.edge_dim is not None

            if expects_edge_dim:
                if current_edge_attr is None:
                    logging.debug(f"GATv2Conv expects edge_attr but none provided. Edge index shape: {edge_index.shape}")
                    current_edge_attr = None
                elif len(current_edge_attr.shape) == 1:
                    current_edge_attr = current_edge_attr.unsqueeze(-1)
            else:
                current_edge_attr = None

            # Pass through GNN layers
            for i, layer in enumerate(self.gnn_layers):
                 layer_expects_edge_dim = hasattr(layer, 'edge_dim') and layer.edge_dim is not None
                 layer_edge_attr = None # Default to None

                 if layer_expects_edge_dim:
                     if current_edge_attr is not None:
                          if len(current_edge_attr.shape) == 1:
                               layer_edge_attr = current_edge_attr.unsqueeze(-1)
                          else:
                               layer_edge_attr = current_edge_attr
                     # else: layer_edge_attr remains None

                 # Check for empty edge_index
                 current_edge_index = edge_index if edge_index.numel() > 0 else torch.empty((2, 0), dtype=torch.long, device=gnn_x.device)

                 try:
                    if layer_expects_edge_dim:
                        gnn_x = layer(gnn_x, current_edge_index, edge_attr=layer_edge_attr)
                    else:
                        gnn_x = layer(gnn_x, current_edge_index)
                 except IndexError as e:
                     logging.error(f"IndexError in GNN layer {i}: {e}. Edge index shape: {current_edge_index.shape}, edge attr shape: {layer_edge_attr.shape if layer_edge_attr is not None else 'None'}. Input node features shape: {gnn_x.shape}. Skipping layer propagation for safety.")
                     gnn_x = lstm_out_last # Revert to LSTM output if GNN fails badly
                     break
                 except Exception as e:
                      logging.error(f"Error in GNN layer {i}: {e}", exc_info=True)
                      gnn_x = lstm_out_last
                      break

                 if isinstance(gnn_x, tuple):
                     gnn_x = gnn_x[0]

                 gnn_x = F.elu(gnn_x)
                 gnn_x = F.dropout(gnn_x, p=self.dropout_rate, training=self.training)
            # gnn_x shape: (batch_size * num_nodes, gnn_hidden_dim * gnn_heads)
        else:
            # If no GNN layers, create an empty tensor for concatenation
            gnn_x = torch.empty((lstm_out_last.shape[0], 0), device=lstm_out_last.device)


        # --- Fusion Path ---
        fused_features = torch.cat([lstm_out_last, gnn_x], dim=1)

        # Check if fusion input matches expected dimension from init
        if fused_features.shape[1] != self.fusion_fc1.in_features:
             logging.error(f"Fusion dimension mismatch! Expected {self.fusion_fc1.in_features}, Got {fused_features.shape[1]}.lstm_out: {lstm_out_last.shape}, gnn_out: {gnn_x.shape}")
             return torch.zeros((fused_features.shape[0], self.fusion_fc2.out_features), device=fused_features.device)


        x = self.fusion_dropout(F.relu(self.fusion_fc1(fused_features)))
        x = self.fusion_fc2(x) # Final output prediction

        return x # Shape: (batch_size * num_nodes, output_dim)


# --- Custom Loss Function ---
class PunishingMSELoss(nn.Module):
    """
    Custom loss function: MSE + alpha * Mean Quartic Error.
    Penalizes larger errors more significantly than standard MSE.
    """
    def __init__(self, alpha=0.1):
        super().__init__()
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        self.alpha = alpha
        self.mse = nn.MSELoss(reduction='none') # Calculate element-wise MSE first

    def forward(self, predictions, targets):
        # Ensure shapes match for element-wise calculation
        if predictions.shape != targets.shape:
            try:
                # Attempt to reshape targets if they have an extra dimension (e.g., [N, M, 1] vs [N, M])
                if len(targets.shape) == len(predictions.shape) + 1 and targets.shape[-1] == 1:
                    targets = targets.squeeze(-1)
                # Attempt reshape only if shapes differ now
                if predictions.shape != targets.shape:
                     targets = targets.view_as(predictions)
            except RuntimeError as e:
                 raise ValueError(f"Shape mismatch in PunishingMSELoss: predictions {predictions.shape}, targets {targets.shape}. Error: {e}")

        error = predictions - targets
        mse_loss_elementwise = self.mse(predictions, targets) # Equivalent to error**2
        quartic_error_elementwise = torch.pow(error, 4)

        # Handle potential NaNs introduced by large errors raised to power 4
        quartic_error_elementwise = torch.nan_to_num(quartic_error_elementwise, nan=1e6) # Replace NaNs with a large number

        mean_mse_loss = torch.mean(mse_loss_elementwise)
        mean_quartic_loss = torch.mean(quartic_error_elementwise)

        if torch.isnan(mean_mse_loss) or torch.isnan(mean_quartic_loss):
            logging.warning(f"NaN detected in loss components: MSE={mean_mse_loss}, Quartic={mean_quartic_loss}. Alpha={self.alpha}")
            # Return a large loss value to penalize this state, avoiding NaN propagation
            return torch.tensor(1e6, device=predictions.device, dtype=predictions.dtype, requires_grad=True)


        combined_loss = mean_mse_loss + self.alpha * mean_quartic_loss

        # Final check for NaN in combined loss
        if torch.isnan(combined_loss):
            logging.error(f"Combined loss is NaN! MSE={mean_mse_loss}, Quartic={mean_quartic_loss}, Alpha={self.alpha}")
            return torch.tensor(1e6, device=predictions.device, dtype=predictions.dtype, requires_grad=True)


        return combined_loss


# --- Global variable for hook (Used only in final evaluation) ---
captured_gnn_output = None
hook_handle = None # Global handle for the hook

def gnn_hook_fn(module, input, output):
    """Forward hook to capture the output of the last GNN layer."""
    global captured_gnn_output
    if isinstance(output, torch.Tensor):
        captured_gnn_output = output.detach().cpu()
    elif isinstance(output, tuple):
        # If GAT returns (out, attn_weights), take the first element
        captured_gnn_output = output[0].detach().cpu()
        logging.debug("Hook captured tuple output, taking first element.")
    else:
        logging.warning(f"Hook captured unexpected output type: {type(output)}")


# --- Training & Evaluation Functions ---
def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    num_samples = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        try:
            out = model(data)
            target = data.y

            # Basic check for NaN/Inf in output/target before loss calc
            if torch.isnan(out).any() or torch.isinf(out).any():
                logging.warning("[Train] NaN/Inf detected in model output. Skipping batch.")
                continue
            if torch.isnan(target).any() or torch.isinf(target).any():
                logging.warning("[Train] NaN/Inf detected in target. Skipping batch.")
                continue


            loss = criterion(out, target)
            if torch.isnan(loss) or torch.isinf(loss):
                logging.warning(f"[Train] NaN/Inf loss detected! Skipping batch.")
                continue

            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * data.num_graphs
            num_samples += data.num_graphs

        except ValueError as e:
            logging.error(f"[Train] ValueError during training step: {e}. Skipping batch.", exc_info=True)
            continue # Skip batch on error
        except RuntimeError as e:
             if "CUDA out of memory" in str(e):
                 logging.error(f"[Train] CUDA OOM Error: {e}. Skipping batch.")
                 # Consider reducing batch size or model size if this happens often
                 # Clear cache might help sometimes, but not always reliable
                 # torch.cuda.empty_cache() # Use with caution, can slow things down
                 continue
             elif "determin" in str(e): # Catch potential cuDNN determinism errors
                 logging.error(f"[Train] CuDNN Determinism Error: {e}. Try setting deterministic=False. Skipping batch.")
                 continue
             else:
                 logging.error(f"[Train] RuntimeError: {e}. Skipping batch.", exc_info=True)
                 continue
        except Exception as e:
             logging.error(f"[Train] Unexpected error: {e}", exc_info=True)
             continue # Skip batch

    return total_loss / num_samples if num_samples > 0 else 0


def evaluate(model, loader, criterion, device, scaler_y=None, num_nodes_per_graph=None, is_final_eval=False, hook_handle_local=None):
    """Modified evaluate function to handle GNN output capture and NaN metrics."""
    model.eval()
    total_loss = 0
    all_preds_scaled_list = []
    all_targets_scaled_list = []
    all_preds_orig_list = []
    all_targets_orig_list = []
    global captured_gnn_output # Use the global variable for capture
    embeddings_collected = False
    num_samples = 0

    # --- Hook activation/deactivation ---
    local_hook_active = False
    if is_final_eval and hook_handle_local is not None and hasattr(model, 'gnn_layers') and len(model.gnn_layers) > 0:
        captured_gnn_output = None
        local_hook_active = True # Flag that we intend to capture
    # ---

    with torch.no_grad():
        for i, data in enumerate(loader):
            data = data.to(device)

            try:
                # Run model forward pass
                out = model(data)

                # If hook is active and this is the first batch, check if capture worked
                if local_hook_active and i == 0 and captured_gnn_output is not None:
                    embeddings_collected = True
                    logging.info(f"Hook captured GNN output of shape: {captured_gnn_output.shape}")
                    local_hook_active = False # Stop checking after first batch


                target = data.y

                # Check for NaNs/Infs before loss calculation
                if torch.isnan(out).any() or torch.isinf(out).any():
                    logging.warning(f"[Eval] NaN/Inf detected in model output (Batch {i}). Skipping loss & metric calc for this batch.")
                    continue
                if torch.isnan(target).any() or torch.isinf(target).any():
                    logging.warning(f"[Eval] NaN/Inf detected in target (Batch {i}). Skipping loss & metric calc for this batch.")
                    continue

                loss = criterion(out, target)
                if torch.isnan(loss) or torch.isinf(loss):
                    logging.warning(f"[Eval] NaN/Inf loss detected (Batch {i})! Skipping loss accumulation.")
                else:
                    total_loss += loss.item() * data.num_graphs
                    num_samples += data.num_graphs

                # Store results (only if loss calc was valid or skip if NaNs detected earlier)
                all_preds_scaled_list.append(out.cpu().numpy())
                all_targets_scaled_list.append(target.cpu().numpy())

                # Inverse transform if possible
                if scaler_y and num_nodes_per_graph is not None and num_nodes_per_graph > 0:
                    batch_size = data.num_graphs
                    # Ensure output dim matches expected before reshape
                    expected_out_dim = 1 # Assuming OUTPUT_DIM is 1
                    if out.shape[-1] != expected_out_dim:
                         logging.warning(f"[Eval] Output dimension mismatch in batch {i}. Expected {expected_out_dim}, got {out.shape[-1]}. Skipping inverse transform.")
                         continue

                    pred_batch_reshaped = out.view(batch_size, num_nodes_per_graph, -1).squeeze(-1).cpu().numpy()
                    target_batch_reshaped = target.view(batch_size, num_nodes_per_graph, -1).squeeze(-1).cpu().numpy()

                    # Check again for NaNs after potential reshape, before inverse transform
                    if not np.isnan(pred_batch_reshaped).any() and not np.isnan(target_batch_reshaped).any():
                        # Ensure scaler expects correct number of features (num_nodes)
                        if pred_batch_reshaped.shape[1] == scaler_y.n_features_in_:
                            pred_rescaled = scaler_y.inverse_transform(pred_batch_reshaped)
                            target_rescaled = scaler_y.inverse_transform(target_batch_reshaped)
                            all_preds_orig_list.append(pred_rescaled)
                            all_targets_orig_list.append(target_rescaled)
                        else:
                            logging.warning(f"[Eval] Scaler feature mismatch in batch {i}. Expected {scaler_y.n_features_in_}, got {pred_batch_reshaped.shape[1]}. Skipping inverse transform.")
                            nan_array = np.full((batch_size, num_nodes_per_graph), np.nan)
                            all_preds_orig_list.append(nan_array)
                            all_targets_orig_list.append(nan_array)
                    else:
                         logging.warning(f"[Eval] NaNs detected in batch {i} preds/targets before inverse transform, skipping.")
                         nan_array = np.full((batch_size, num_nodes_per_graph), np.nan)
                         all_preds_orig_list.append(nan_array)
                         all_targets_orig_list.append(nan_array)

            except ValueError as e:
                logging.error(f"[Eval] ValueError during eval step (Batch {i}): {e}.", exc_info=True)
                continue
            except RuntimeError as e:
                 logging.error(f"[Eval] RuntimeError during eval step (Batch {i}): {e}.", exc_info=True)
                 continue
            except Exception as e:
                logging.error(f"[Eval] Unexpected error during eval step (Batch {i}): {e}", exc_info=True)
                continue

    # Post-loop calculations
    avg_loss = total_loss / num_samples if num_samples > 0 else float('inf') # Return Inf if no samples processed
    final_preds_orig = np.nan
    final_targets_orig = np.nan
    rmse, mae = float('nan'), float('nan') # Default to NaN

    if all_preds_orig_list and all_targets_orig_list:
        try:
            # Filter out any potential non-array entries or all-NaN arrays if added as placeholders
            valid_preds_orig = [arr for arr in all_preds_orig_list if isinstance(arr, np.ndarray) and not np.all(np.isnan(arr))]
            valid_targets_orig = [arr for arr in all_targets_orig_list if isinstance(arr, np.ndarray) and not np.all(np.isnan(arr))]

            if valid_preds_orig and valid_targets_orig: # Ensure we have valid data to concat
                final_preds_orig = np.concatenate(valid_preds_orig)
                final_targets_orig = np.concatenate(valid_targets_orig)

                # Calculate metrics only on non-NaN pairs
                valid_mask = ~np.isnan(final_preds_orig) & ~np.isnan(final_targets_orig)
                if np.sum(valid_mask) > 0:
                    targets_flat_valid = final_targets_orig[valid_mask].flatten()
                    preds_flat_valid = final_preds_orig[valid_mask].flatten()

                    if len(targets_flat_valid) == len(preds_flat_valid) and len(targets_flat_valid) > 0:
                         rmse = np.sqrt(mean_squared_error(targets_flat_valid, preds_flat_valid))
                         mae = mean_absolute_error(targets_flat_valid, preds_flat_valid)
                         # Final check if metrics calculation resulted in NaN (e.g., due to overflow with large numbers)
                         if np.isnan(rmse): rmse = float('inf')
                         if np.isnan(mae): mae = float('inf')
                    else:
                        logging.warning("Valid mask applied, but resulting arrays are empty or mismatched.")
                        rmse, mae = float('inf'), float('inf')

                else:
                    logging.warning("No valid (non-NaN) original scale preds/targets pairs found after concatenation.")
                    rmse, mae = float('inf'), float('inf') # Indicate failure
            else:
                logging.warning("No valid original scale prediction/target batches to concatenate.")
                rmse, mae = float('inf'), float('inf') # Indicate failure

        except ValueError as e:
             logging.error(f"Error concatenating/calculating metrics: {e}. Check shapes. Pred shapes: {[a.shape for a in valid_preds_orig]}, Target shapes: {[a.shape for a in valid_targets_orig]}", exc_info=True)
             rmse, mae = float('inf'), float('inf')
        except Exception as e:
             logging.error(f"Unexpected error during metric calculation: {e}", exc_info=True)
             rmse, mae = float('inf'), float('inf')
    else:
        logging.warning("Original scale prediction/target lists are empty.")
        rmse, mae = float('inf'), float('inf')

    # Ensure we return float('inf') instead of numpy NaN for the primary metrics
    if np.isnan(avg_loss): avg_loss = float('inf')
    if np.isnan(rmse): rmse = float('inf')
    if np.isnan(mae): mae = float('inf')

    # Return captured embeddings only if final eval and capture was successful
    final_embeddings = captured_gnn_output if is_final_eval and embeddings_collected else None

    # If not final eval, return only metrics
    if not is_final_eval:
         return avg_loss, rmse, mae
    else:
         # Ensure final arrays are returned even if metrics calculation failed
         final_preds_orig_ret = final_preds_orig if 'final_preds_orig' in locals() and isinstance(final_preds_orig, np.ndarray) else np.array([])
         final_targets_orig_ret = final_targets_orig if 'final_targets_orig' in locals() and isinstance(final_targets_orig, np.ndarray) else np.array([])
         return avg_loss, rmse, mae, final_preds_orig_ret, final_targets_orig_ret, final_embeddings


# --- Wrapper Module for Tracing (Used only in final evaluation if needed) ---
class GraphTracerWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        # Check if the model's GNN layers expect edge_dim
        self.expects_edge_dim = False
        if hasattr(model, 'gnn_layers') and len(model.gnn_layers) > 0:
             try:
                 first_gnn_layer = model.gnn_layers[0]
                 sig = inspect.signature(first_gnn_layer.forward) # Check forward signature
                 self.expects_edge_dim = 'edge_attr' in sig.parameters
                 # Also check if the layer itself was initialized expecting edge_dim
                 if not self.expects_edge_dim and hasattr(first_gnn_layer, 'edge_dim') and first_gnn_layer.edge_dim is not None:
                     self.expects_edge_dim = True

             except Exception as e:
                 logging.warning(f"Could not inspect GNN layer signature to determine edge_attr usage: {e}")
                 # Default assumption based on GATv2Conv init might be needed if forward inspect fails
                 try:
                     sig_init = inspect.signature(GATv2Conv.__init__)
                     self.expects_edge_dim = 'edge_dim' in sig_init.parameters and hasattr(first_gnn_layer, 'edge_dim') and first_gnn_layer.edge_dim is not None
                 except Exception:
                     self.expects_edge_dim = False # Fallback

    def forward(self, x_seq, edge_index, edge_attr=None):
        # Reconstruct a minimal Data object suitable for tracing
        num_nodes = x_seq.shape[0] # Total nodes in the batch being traced
        dummy_batch = torch.zeros(num_nodes, dtype=torch.long, device=x_seq.device)

        # Handle edge_attr based on whether it's provided AND expected by the model
        if not self.expects_edge_dim:
            edge_attr_to_pass = None # Don't pass edge_attr if not expected
        else:
             # If edge_attr is expected, but None is passed, pass None
             # If edge_attr is provided, pass it
             edge_attr_to_pass = edge_attr
             # Ensure it has the feature dimension if required (e.g., [num_edges, 1])
             if edge_attr_to_pass is not None and len(edge_attr_to_pass.shape) == 1:
                 edge_attr_to_pass = edge_attr_to_pass.unsqueeze(-1)


        # Create a Data object
        data = Data(x_seq=x_seq,
                    edge_index=edge_index,
                    edge_attr=edge_attr_to_pass,
                    batch=dummy_batch, # Include dummy batch
                    num_nodes=num_nodes) # Include num_nodes

        # Call the original model's forward method
        return self.model(data)
# --- END Wrapper Module ---


# --- Optuna Objective Function ---
def objective(trial: optuna.trial.Trial,
              X_train_seq, y_train_seq,
              X_val_seq, y_val_seq,
              ticker_order, train_data_df, # Pass train df for graph building
              scaler_y, device, num_features, num_nodes,
              base_log_dir):
    """Optuna objective function for hyperparameter tuning."""
    trial_start_time = time.time() # <<< ADDED: Start timer for trial

    # --- TensorBoard Setup for Trial ---
    trial_log_dir = os.path.join(base_log_dir, f"trial_{trial.number}")
    writer = SummaryWriter(log_dir=trial_log_dir)
    logging.info(f"Trial {trial.number}: Starting... Logging to {trial_log_dir}")

    # --- Hyperparameter Suggestions ---
    try:
        # Graph Params
        correlation_threshold = trial.suggest_float("correlation_threshold", 0.1, 0.6)
        correlation_window = trial.suggest_int("correlation_window", 30, 100, step=10)
        # Model Params
        lstm_hidden_dim = trial.suggest_categorical("lstm_hidden_dim", [32, 64, 128, 192])
        lstm_layers = trial.suggest_int("lstm_layers", 1, 3)
        gnn_hidden_dim = trial.suggest_categorical("gnn_hidden_dim", [32, 64, 96, 128])
        # Allow 0 GNN layers
        gnn_layers = trial.suggest_int("gnn_layers", 0, 3) # <<< MODIFIED: Allow 0 layers
        # Only suggest heads if layers > 0
        gnn_heads = trial.suggest_categorical("gnn_heads", [2, 4, 8]) if gnn_layers > 0 else 1 # Default to 1 if no layers

        fusion_hidden_dim = trial.suggest_categorical("fusion_hidden_dim", [64, 96, 128, 192])
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
        # Training Params
        learning_rate = trial.suggest_float("learning_rate", 5e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        loss_alpha = trial.suggest_float("loss_alpha", 0.0, 0.25)

    except Exception as e:
         logging.error(f"Trial {trial.number}: Error suggesting hyperparameters: {e}")
         if writer: writer.close()
         return float('inf') # Return high value if suggestion fails

    # --- Build Graph for Trial ---
    try:
        edge_index, edge_weight = build_correlation_graph(train_data_df, ticker_order, correlation_window, correlation_threshold, TARGET)
        edge_index = edge_index.to(device)
        if edge_weight is not None and edge_weight.numel() > 0:
            edge_weight = edge_weight.to(device)
        else:
            edge_weight = None # Important to set to None if empty
    except Exception as e:
        logging.error(f"Trial {trial.number}: Error building graph: {e}", exc_info=True)
        if writer: writer.close()
        return float('inf') # Return high value if graph build fails

    # --- Create Datasets and DataLoaders for Trial ---
    try:
        train_dataset = StockDataset(X_train_seq, y_train_seq, edge_index, edge_weight)
        # Validation dataset might be empty if split is 0 or data too short
        val_dataset = StockDataset(X_val_seq, y_val_seq, edge_index, edge_weight) if X_val_seq is not None and X_val_seq.size > 0 else None

        # Ensure batch size is not larger than dataset size
        train_batch_size_actual = min(batch_size, len(train_dataset)) if len(train_dataset) > 0 else 1
        val_batch_size_actual = min(batch_size, len(val_dataset)) if val_dataset and len(val_dataset) > 0 else 1

        if train_batch_size_actual == 0:
            logging.error(f"Trial {trial.number}: Train dataset empty or invalid batch size.")
            if writer: writer.close()
            return float('inf')

        # Use drop_last=True for training to ensure consistent batch shapes if possible
        # Only create loader if dataset has samples
        train_loader = DataLoader(train_dataset, batch_size=train_batch_size_actual, shuffle=True, drop_last=True) if len(train_dataset) > 0 else None
        val_loader = DataLoader(val_dataset, batch_size=val_batch_size_actual, shuffle=False) if val_dataset and len(val_dataset) > 0 and val_batch_size_actual > 0 else None

        if not train_loader:
             logging.error(f"Trial {trial.number}: No training loader created.")
             if writer: writer.close()
             return float('inf')

        if not val_loader:
             logging.warning(f"Trial {trial.number}: No validation loader created (Val dataset size: {len(val_dataset) if val_dataset else 0}). Cannot perform validation or pruning.")
             # If no validation is possible, Optuna cannot prune or optimize based on validation.
             # Decide behavior: run fixed epochs and return train loss? Or skip trial? Let's skip.
             if writer: writer.close()
             return float('inf') # Cannot optimize without validation metric

    except Exception as e:
        logging.error(f"Trial {trial.number}: Error creating datasets/loaders: {e}", exc_info=True)
        if writer: writer.close()
        return float('inf')

    # --- Initialize Model, Optimizer, Loss for Trial ---
    try:
        model = StockLSTM_GNN(
            num_features=num_features,
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_layers=lstm_layers,
            gnn_hidden_dim=gnn_hidden_dim,
            gnn_layers=gnn_layers,
            gnn_heads=gnn_heads,
            fusion_hidden_dim=fusion_hidden_dim,
            output_dim=OUTPUT_DIM,
            dropout_rate=dropout_rate
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = PunishingMSELoss(alpha=loss_alpha).to(device)
    except ValueError as e:
         logging.error(f"Trial {trial.number}: Error initializing model (likely invalid dimensions from hyperparameters): {e}", exc_info=True)
         if writer: writer.close()
         return float('inf') # Penalize invalid hyperparameter combinations
    except Exception as e:
        logging.error(f"Trial {trial.number}: Error initializing model/optimizer/loss: {e}", exc_info=True)
        if writer: writer.close()
        return float('inf')

    # --- Training Loop for Trial ---
    best_trial_metric = float('inf') # Track the best metric *within this trial*

    try:
        for epoch in range(1, EPOCHS + 1):
            train_loss = train(model, train_loader, optimizer, criterion, device)
            val_loss, val_rmse, val_mae = evaluate(model, val_loader, criterion, device, scaler_y, num_nodes)

            # Log metrics to TensorBoard for the trial
            if writer:
                writer.add_scalar('Loss/Train', train_loss, epoch)
                if not np.isinf(val_loss) and not np.isnan(val_loss): writer.add_scalar('Loss/Validation', val_loss, epoch)
                if not np.isinf(val_rmse) and not np.isnan(val_rmse): writer.add_scalar('RMSE/Validation', val_rmse, epoch)
                if not np.isinf(val_mae) and not np.isnan(val_mae): writer.add_scalar('MAE/Validation', val_mae, epoch)

            logging.debug(f'Trial {trial.number} - Epoch: {epoch:03d}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Val RMSE: {val_rmse:.4f}, Val MAE: {val_mae:.4f}')

            # --- Pruning & Metric Reporting ---
            # Select the metric to report and potentially update best
            if METRIC_TO_OPTIMIZE == "val_loss":
                current_metric = val_loss
            elif METRIC_TO_OPTIMIZE == "val_rmse":
                current_metric = val_rmse
            else: # Default to val_loss if metric name is wrong
                current_metric = val_loss

            # Handle NaN/Inf metric values before reporting/comparing
            if np.isnan(current_metric) or np.isinf(current_metric):
                logging.warning(f"Trial {trial.number} - Epoch {epoch}: Invalid metric ({METRIC_TO_OPTIMIZE}={current_metric}). Treating as high value for pruning/best tracking.")
                current_metric = float('inf') # Treat invalid as very bad

            # Update best metric for this trial
            best_trial_metric = min(best_trial_metric, current_metric)

            if PRUNING_ENABLED:
                trial.report(current_metric, epoch)
                if trial.should_prune():
                    trial_duration = time.time() - trial_start_time # <<< ADDED: Calculate duration
                    logging.info(f"Trial {trial.number} pruned at epoch {epoch}. Duration: {trial_duration:.2f} seconds.")
                    if writer: writer.close()
                    raise TrialPruned()

        # --- End of Epoch Loop ---
        trial_duration = time.time() - trial_start_time # <<< ADDED: Calculate duration
        if writer: writer.close() # Close writer at the end of a successful trial
        logging.info(f"Trial {trial.number} finished. Best validation metric ({METRIC_TO_OPTIMIZE}): {best_trial_metric:.6f}. Duration: {trial_duration:.2f} seconds.")
        return best_trial_metric # Return the best validation metric achieved during the trial

    except TrialPruned as e:
         # Need to re-raise the exception for Optuna to handle pruning correctly
         if writer: writer.close()
         raise e
    except Exception as e:
         trial_duration = time.time() - trial_start_time # <<< ADDED: Calculate duration even on error
         logging.error(f"Trial {trial.number}: Error during training/evaluation loop: {e}. Duration: {trial_duration:.2f} seconds.", exc_info=True)
         if writer: writer.close()
         return float('inf') # Return high value if trial crashes


# --- Optuna Callback for TQDM Progress Bar ---
trial_times = [] # Store trial durations
def tqdm_callback(study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
    """Optuna callback to update tqdm progress bar and show trial duration."""
    pbar.update(1)
    if trial.state == optuna.trial.TrialState.COMPLETE:
        duration = (trial.datetime_complete - trial.datetime_start).total_seconds()
        trial_times.append(duration)
        avg_time = np.mean(trial_times) if trial_times else 0
        # Use study.best_value which is guaranteed to exist if at least one trial completed
        best_value_str = f"{study.best_value:.4f}" if study.best_value is not None else "N/A"
        pbar.set_postfix_str(f"Last trial: {duration:.1f}s, Avg: {avg_time:.1f}s, Best: {best_value_str}")
    elif trial.state == optuna.trial.TrialState.PRUNED:
        duration = (trial.datetime_complete - trial.datetime_start).total_seconds()
        trial_times.append(duration)
        avg_time = np.mean(trial_times) if trial_times else 0
        best_value_str = f"{study.best_value:.4f}" if study.best_value is not None else "N/A"
        pbar.set_postfix_str(f"Last trial: {duration:.1f}s (Pruned), Avg: {avg_time:.1f}s, Best: {best_value_str}")
    elif trial.state == optuna.trial.TrialState.FAIL:
        duration = (trial.datetime_complete - trial.datetime_start).total_seconds() if trial.datetime_complete else -1
        if duration >= 0: trial_times.append(duration) # Only append valid durations
        avg_time = np.mean([t for t in trial_times if t >= 0]) if any(t >= 0 for t in trial_times) else 0
        best_value_str = f"{study.best_value:.4f}" if study.best_value is not None else "N/A"
        pbar.set_postfix_str(f"Last trial: FAILED ({duration:.1f}s), Avg: {avg_time:.1f}s, Best: {best_value_str}")


# --- Main Execution ---
if __name__ == "__main__":
    logging.info("--- Starting Optuna LSTM-GNN Stock Forecasting ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # --- Pre-computation: Load and Prepare Data ONCE ---
    try:
        combined_df, ticker_order = load_and_align_data(DATA_FOLDER, TICKERS)
        num_nodes = len(ticker_order)
        num_features = len(FEATURES)
        if num_nodes == 0: raise ValueError("No tickers remaining after data loading.")
        logging.info(f"Data loaded for {num_nodes} tickers: {ticker_order}")

        data_values = combined_df.values
        total_samples = len(combined_df)
        if total_samples <= SEQUENCE_LENGTH + PREDICTION_HORIZON:
             raise ValueError(f"Insufficient data ({total_samples}) for sequence ({SEQUENCE_LENGTH}) + horizon ({PREDICTION_HORIZON}).")

        train_end_idx = int(total_samples * TRAIN_SPLIT)
        val_end_idx = train_end_idx + int(total_samples * VALIDATION_SPLIT)

        train_data_df = combined_df.iloc[:train_end_idx] # Keep train df for correlation calc later
        val_data_df = combined_df.iloc[train_end_idx:val_end_idx]
        test_data_df = combined_df.iloc[val_end_idx:]

        if len(train_data_df) == 0 or len(test_data_df) == 0:
             raise ValueError("Train or Test data split is empty.")
        if len(val_data_df) == 0 and VALIDATION_SPLIT > 0:
             logging.warning("Validation split > 0, but validation data is empty.")

        logging.info(f"Data split: Train {len(train_data_df)}, Validation {len(val_data_df)}, Test {len(test_data_df)}")

        # Scale features (X) based on training data
        scaler_x = MinMaxScaler()
        train_scaled_x = scaler_x.fit_transform(train_data_df.values)
        val_scaled_x = scaler_x.transform(val_data_df.values) if len(val_data_df) > 0 else np.array([]) # Use empty array instead of None
        test_scaled_x = scaler_x.transform(test_data_df.values)

        # Scale target (Y) based on training data
        target_cols = [f"{ticker}_{TARGET}" for ticker in ticker_order]
        scaler_y = MinMaxScaler() # Scaler for the target variable ('Close')
        train_scaled_y = scaler_y.fit_transform(train_data_df[target_cols].values)
        val_scaled_y = scaler_y.transform(val_data_df[target_cols].values) if len(val_data_df) > 0 else np.array([]) # Use empty array instead of None
        test_scaled_y = scaler_y.transform(test_data_df[target_cols].values)

        # Create sequences (inputs and targets)
        X_train_seq_flat, _ = create_sequences(train_scaled_x, SEQUENCE_LENGTH, PREDICTION_HORIZON)
        _, y_train_seq = create_sequences(train_scaled_y, SEQUENCE_LENGTH, PREDICTION_HORIZON)

        X_val_seq_flat, _ = create_sequences(val_scaled_x, SEQUENCE_LENGTH, PREDICTION_HORIZON) # Handles empty val_scaled_x
        _, y_val_seq = create_sequences(val_scaled_y, SEQUENCE_LENGTH, PREDICTION_HORIZON) # Handles empty val_scaled_y

        X_test_seq_flat, _ = create_sequences(test_scaled_x, SEQUENCE_LENGTH, PREDICTION_HORIZON)
        _, y_test_seq = create_sequences(test_scaled_y, SEQUENCE_LENGTH, PREDICTION_HORIZON)

        # Validate sequence creation
        if X_train_seq_flat.size == 0 or y_train_seq.size == 0: raise ValueError("Training sequences are empty.")
        if X_test_seq_flat.size == 0 or y_test_seq.size == 0: raise ValueError("Test sequences are empty.")
        if X_val_seq_flat.size == 0 and len(val_data_df) > 0:
             logging.warning("Validation sequences are empty despite non-empty validation data. Check val data length vs sequence/horizon.")


        # Reshape X sequences: (num_samples, seq_len, num_nodes * num_features) -> (num_samples, num_nodes, seq_len, num_features)
        def reshape_sequence_data(X_seq_flat, num_nodes, num_features):
            if X_seq_flat is None or X_seq_flat.size == 0: return None # Return None if input is empty
            num_samples, seq_len, flat_features = X_seq_flat.shape
            expected_flat = num_nodes * num_features
            if flat_features != expected_flat:
                 raise ValueError(f"Reshape failed: Expected {expected_flat} flat features, got {flat_features}.")
            # Reshape and transpose: (samples, seq, nodes*feat) -> (samples, seq, nodes, feat) -> (samples, nodes, seq, feat)
            return X_seq_flat.reshape(num_samples, seq_len, num_nodes, num_features).transpose(0, 2, 1, 3)

        X_train_seq = reshape_sequence_data(X_train_seq_flat, num_nodes, num_features)
        X_val_seq = reshape_sequence_data(X_val_seq_flat, num_nodes, num_features) # Will be None if X_val_seq_flat is empty
        X_test_seq = reshape_sequence_data(X_test_seq_flat, num_nodes, num_features)

        logging.info("Data loading and preprocessing complete.")

    except (FileNotFoundError, ValueError) as e:
        logging.error(f"Failed during initial data preparation: {e}", exc_info=True)
        exit(1)
    except Exception as e:
        logging.error(f"Unexpected error during initial data preparation: {e}", exc_info=True)
        exit(1)

    # --- Optuna Study ---
    study = None # Initialize study variable
    best_trial = None # Initialize best_trial variable
    try:
        # Define study direction based on metric
        direction = "minimize" # Both loss and RMSE should be minimized

        # Use a pruner if enabled
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=EPOCHS // 3) if PRUNING_ENABLED else optuna.pruners.NopPruner()

        # <<< MODIFIED: Setup persistent storage >>>
        # Create the base TensorBoard log directory FIRST
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        study_log_dir = os.path.join(TENSORBOARD_LOG_DIR_BASE, current_time + "_study")
        os.makedirs(study_log_dir, exist_ok=True)
        logging.info(f"Optuna study logs (individual trials) will be under: {study_log_dir}")

        # Define study name and database path (within the study log dir)
        study_name = f"gnn_lstm_study_{current_time}" # Unique name for the study
        db_filename = "optuna_study.db"
        db_path = os.path.join(study_log_dir, db_filename)
        # Ensure the path is absolute or correctly relative for SQLite
        db_path_abs = os.path.abspath(db_path)
        storage_url = f"sqlite:///{db_path_abs}" # SQLite database URL requires absolute path or careful relative path handling

        logging.info(f"Creating/Loading Optuna study '{study_name}' using storage: {storage_url}")

        study = optuna.create_study(
            study_name=study_name,
            storage=storage_url, # Specify the storage URL
            direction=direction,
            pruner=pruner,
            load_if_exists=True # <<< IMPORTANT: Set to True to resume if the study/db exists
        )
        # <<< END MODIFIED SECTION >>>

        # Check if we are resuming and adjust N_TRIALS if needed
        # Filter trials carefully by state to count completed/pruned ones
        completed_trial_states = [optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED]
        completed_trials = len([t for t in study.trials if t.state in completed_trial_states])

        remaining_trials = N_TRIALS - completed_trials
        if remaining_trials <= 0 and N_TRIALS > 0:
             logging.warning(f"Study '{study_name}' already has {completed_trials} completed/pruned trials. Requested N_TRIALS is {N_TRIALS}. No new trials will run unless N_TRIALS is increased.")
             # Decide if you want to exit or proceed (e.g., just run the final evaluation)
             # For now, let's proceed to potentially run the final eval based on existing best trial
        elif completed_trials > 0:
             logging.info(f"Resuming study '{study_name}'. {completed_trials} trials already done. Running up to {remaining_trials} more trials.")


        # <<< MODIFIED: Wrap optimize call with tqdm >>>
        logging.info(f"--- Starting/Resuming Optuna Optimization ({remaining_trials if remaining_trials > 0 else 0} trials remaining, timeout={OPTUNA_TIMEOUT}s) ---")
        global pbar # Make pbar global so callback can access it
        # Adjust tqdm total if resuming
        with tqdm(total=N_TRIALS, initial=completed_trials, desc="Optimizing Hyperparameters") as pbar:
            # Only run optimize if there are trials left to run
            if remaining_trials > 0:
                study.optimize(
                    lambda trial: objective( # Use lambda to pass arguments to objective
                        trial,
                        X_train_seq, y_train_seq,
                        X_val_seq, y_val_seq,
                        ticker_order, train_data_df, # Pass original train DF for graph
                        scaler_y, device, num_features, num_nodes,
                        study_log_dir # Pass base log dir for trials
                    ),
                    n_trials=remaining_trials, # Run only the remaining number of trials
                    timeout=OPTUNA_TIMEOUT, # Timeout still applies overall
                    callbacks=[tqdm_callback], # Pass the callback
                    # catch=(Exception,) # Catch generic exceptions within Optuna optimize loop if needed (Objective should handle its own errors)
                )
            else:
                 # If no trials left, manually update pbar if needed or just skip optimize
                 pbar.n = pbar.total # Show bar as full if resuming a completed study
                 pbar.refresh()
                 logging.info("Optimization phase skipped as all trials are complete.")
        # <<< END MODIFIED SECTION >>>

        logging.info(f"Optuna study finished. Total trials in study: {len(study.trials)}")
        # Check if study actually found a best trial (could be empty if all failed/pruned early)
        # Accessing study.best_trial raises error if no trials completed successfully
        valid_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if valid_trials:
            best_trial = study.best_trial
            logging.info(f"Best trial number: {best_trial.number}")
            logging.info(f"Best validation metric ({METRIC_TO_OPTIMIZE}): {best_trial.value:.6f}")
            logging.info("Best hyperparameters:")
            for key, value in best_trial.params.items():
                logging.info(f"  {key}: {value}")
        else:
            logging.warning("Optuna study finished, but no trials completed successfully (all may have failed or been pruned).")
            # Attempt to load again in case the study object wasn't updated but DB has data
            try:
                 logging.info(f"Attempting to reload study '{study_name}' from storage: {storage_url}")
                 study = optuna.load_study(study_name=study_name, storage=storage_url)
                 valid_trials_reload = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
                 if valid_trials_reload:
                      best_trial = study.best_trial
                      logging.info("Loaded best trial from existing study storage.")
                      logging.info(f"Best trial number: {best_trial.number}")
                      logging.info(f"Best validation metric ({METRIC_TO_OPTIMIZE}): {best_trial.value:.6f}")
                      logging.info("Best hyperparameters:")
                      for key, value in best_trial.params.items():
                          logging.info(f"  {key}: {value}")
                 else:
                      logging.error("Still no successfully completed trials found after attempting to reload study. Cannot proceed to final run.")
                      exit(1) # Exit if no successful trial completed or loaded
            except Exception as load_err:
                 logging.error(f"Failed to load study '{study_name}' after optimization finished with no best trial: {load_err}")
                 exit(1)


    except Exception as e:
        logging.error(f"An error occurred during the Optuna study setup or execution: {e}", exc_info=True)
        exit(1)

    # --- Final Run with Best Hyperparameters ---
    # Ensure best_trial is available before proceeding
    if best_trial is None:
        logging.error("Cannot proceed to final run: best_trial is None.")
        exit(1)

    logging.info("--- Starting Final Run with Best Hyperparameters ---")
    writer = None # Initialize writer
    try:
        # Get best params (already checked if best_trial exists)
        best_params = best_trial.params
        # Use the same study_log_dir established earlier
        final_log_dir = os.path.join(study_log_dir, "final_best_run")
        writer = SummaryWriter(log_dir=final_log_dir)
        logging.info(f"Final run logs will be saved to: {final_log_dir}")


        # Build final graph using best params
        final_edge_index, final_edge_weight = build_correlation_graph(
            train_data_df, ticker_order, best_params['correlation_window'], best_params['correlation_threshold'], TARGET
        )
        final_edge_index = final_edge_index.to(device)
        if final_edge_weight is not None and final_edge_weight.numel() > 0:
            final_edge_weight = final_edge_weight.to(device)
        else:
            final_edge_weight = None

        # Create final datasets and dataloaders
        final_train_dataset = StockDataset(X_train_seq, y_train_seq, final_edge_index, final_edge_weight)
        final_val_dataset = StockDataset(X_val_seq, y_val_seq, final_edge_index, final_edge_weight) if X_val_seq is not None and X_val_seq.size > 0 else None
        final_test_dataset = StockDataset(X_test_seq, y_test_seq, final_edge_index, final_edge_weight)

        final_batch_size = min(best_params['batch_size'], len(final_train_dataset)) if len(final_train_dataset)>0 else 1
        final_val_batch_size = min(best_params['batch_size'], len(final_val_dataset)) if final_val_dataset and len(final_val_dataset)>0 else 1
        final_test_batch_size = min(best_params['batch_size'], len(final_test_dataset)) if len(final_test_dataset)>0 else 1

        if final_batch_size == 0 or final_test_batch_size == 0:
             raise ValueError("Final train or test dataset is empty or batch size invalid.")

        final_train_loader = DataLoader(final_train_dataset, batch_size=final_batch_size, shuffle=True, drop_last=True) if len(final_train_dataset) > 0 else None
        final_val_loader = DataLoader(final_val_dataset, batch_size=final_val_batch_size, shuffle=False) if final_val_dataset and final_val_batch_size > 0 else None
        final_test_loader = DataLoader(final_test_dataset, batch_size=final_test_batch_size, shuffle=False) if len(final_test_dataset) > 0 else None

        if not final_train_loader: raise ValueError("Final training loader could not be created.")
        if not final_test_loader: raise ValueError("Final test loader could not be created.")


        # Initialize final model
        final_model = StockLSTM_GNN(
            num_features=num_features,
            lstm_hidden_dim=best_params['lstm_hidden_dim'],
            lstm_layers=best_params['lstm_layers'],
            gnn_hidden_dim=best_params['gnn_hidden_dim'],
            gnn_layers=best_params['gnn_layers'],
            gnn_heads=best_params.get('gnn_heads', 1), # Use .get() for safety if gnn_layers=0
            fusion_hidden_dim=best_params['fusion_hidden_dim'],
            output_dim=OUTPUT_DIM,
            dropout_rate=best_params['dropout_rate']
        ).to(device)

        final_optimizer = optim.Adam(final_model.parameters(), lr=best_params['learning_rate'])
        final_criterion = PunishingMSELoss(alpha=best_params['loss_alpha']).to(device)

        logging.info("Final Model Architecture:")
        print(final_model)
        logging.info(f"Number of parameters: {sum(p.numel() for p in final_model.parameters() if p.requires_grad)}")


        # --- Register Hook for Final Evaluation ---
        # Only register if GNN layers exist in the best model
        if hasattr(final_model, 'gnn_layers') and len(final_model.gnn_layers) > 0:
             try:
                 hook_handle = final_model.gnn_layers[-1].register_forward_hook(gnn_hook_fn)
                 logging.info(f"Registered forward hook on final model GNN layer: {final_model.gnn_layers[-1]}")
             except Exception as e:
                 logging.error(f"Failed to register forward hook on final model: {e}", exc_info=True)
                 hook_handle = None
        else:
             logging.info("No GNN layers in the final model (best_params['gnn_layers']==0). Skipping hook registration.")
             hook_handle = None
        # ---------------------------------------------------------

        # --- Log Model Graph for Final Model ---
        if writer:
            try:
                # Ensure test loader is not empty
                if final_test_loader and len(final_test_loader) > 0:
                    sample_data_batch = next(iter(final_test_loader)).to(device)
                    x_seq_sample = sample_data_batch.x_seq
                    edge_index_sample = sample_data_batch.edge_index
                    edge_attr_sample = sample_data_batch.edge_attr # Can be None

                    model_wrapper = GraphTracerWrapper(final_model).to(device)
                    model_wrapper.eval()
                    graph_input_tuple = (x_seq_sample, edge_index_sample, edge_attr_sample)
                    writer.add_graph(model_wrapper, graph_input_tuple)
                    logging.info("Logged final model graph to TensorBoard.")
                    del model_wrapper, sample_data_batch, x_seq_sample, edge_index_sample, edge_attr_sample # Clean up
                else:
                    logging.warning("Final test loader is empty or None, cannot get sample batch to log model graph.")
            except StopIteration:
                 logging.warning("Could not get sample batch from final test loader to log model graph (StopIteration).")
            except Exception as e:
                logging.error(f"Could not log final model graph to TensorBoard: {e}", exc_info=True)
        # --- END Log Model Graph ---


        # --- Re-Train Final Model ---
        logging.info("--- Re-training final model with best parameters ---")
        # Could potentially train for more epochs here if desired
        final_epochs = EPOCHS # Use the same number of epochs as in trials, or define a different number
        best_final_val_loss = float('inf')
        last_epoch_final = 0

        for epoch in range(1, final_epochs + 1):
            last_epoch_final = epoch
            train_loss = train(final_model, final_train_loader, final_optimizer, final_criterion, device)
            if writer: writer.add_scalar('FinalRun/Loss/Train', train_loss, epoch)

            val_loss, val_rmse, val_mae = float('inf'), float('inf'), float('inf')
            if final_val_loader:
                 val_loss, val_rmse, val_mae = evaluate(final_model, final_val_loader, final_criterion, device, scaler_y, num_nodes)
                 if writer:
                     if not np.isinf(val_loss) and not np.isnan(val_loss): writer.add_scalar('FinalRun/Loss/Validation', val_loss, epoch)
                     if not np.isinf(val_rmse) and not np.isnan(val_rmse): writer.add_scalar('FinalRun/RMSE/Validation', val_rmse, epoch)
                     if not np.isinf(val_mae) and not np.isnan(val_mae): writer.add_scalar('FinalRun/MAE/Validation', val_mae, epoch)

            logging.info(f'Final Run - Epoch: {epoch:03d}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Val RMSE: {val_rmse:.4f}, Val MAE: {val_mae:.4f}')

            # Optional: Save the best model based on validation loss during the final training run
            current_val_loss = val_loss if final_val_loader and not np.isnan(val_loss) and not np.isinf(val_loss) else float('inf')
            if current_val_loss < best_final_val_loss:
                 best_final_val_loss = current_val_loss
                 try:
                     save_path = os.path.join(final_log_dir, 'best_final_model.pth')
                     torch.save(final_model.state_dict(), save_path)
                     logging.info(f"Final run validation loss improved to {best_final_val_loss:.6f}. Saved best final model to {save_path}")
                 except Exception as e:
                     logging.error(f"Error saving best final model: {e}", exc_info=True)

        logging.info("--- Final Model Training Finished ---")

        # --- Final Test Set Evaluation ---
        # Load the best saved final model if it exists
        best_final_model_path = os.path.join(final_log_dir, 'best_final_model.pth')
        if os.path.exists(best_final_model_path):
             try:
                 final_model.load_state_dict(torch.load(best_final_model_path, map_location=device))
                 logging.info(f"Loaded best final model state from '{best_final_model_path}' for test evaluation.")
             except Exception as e:
                 logging.error(f"Error loading best final model state dict: {e}. Evaluating with model state from end of training.", exc_info=True)
        else:
             logging.warning("Best final model state file not found. Evaluating with model state from end of training.")


        test_loss, test_rmse, test_mae, test_preds_orig, test_targets_orig, final_gnn_embeddings = evaluate(
            final_model, final_test_loader, final_criterion, device, scaler_y, num_nodes, is_final_eval=True, hook_handle_local=hook_handle # Pass hook handle
        )
        logging.info(f'--- Final Test Set Evaluation ---')
        logging.info(f'Test Loss (Custom Scaled): {test_loss:.6f}')
        logging.info(f'Test RMSE (Original Scale): {test_rmse:.4f}')
        logging.info(f'Test MAE (Original Scale): {test_mae:.4f}')

        # Log final test metrics to TensorBoard HParams
        final_metrics_dict = {
            'hparam/final_test_loss': test_loss if not np.isnan(test_loss) and not np.isinf(test_loss) else -999.0,
            'hparam/final_test_rmse': test_rmse if not np.isnan(test_rmse) and not np.isinf(test_rmse) else -999.0,
            'hparam/final_test_mae': test_mae if not np.isnan(test_mae) and not np.isinf(test_mae) else -999.0,
        }

        # --- Log Embeddings (Projector) to TensorBoard ---
        if writer and final_gnn_embeddings is not None:
            logging.info(f"Preparing GNN output embeddings for TensorBoard Projector (shape: {final_gnn_embeddings.shape})")
            num_embeddings_captured = final_gnn_embeddings.shape[0]

            if num_embeddings_captured > 0 and num_nodes > 0 and len(ticker_order) == num_nodes:
                # Check if the number of embeddings is a multiple of num_nodes
                if num_embeddings_captured % num_nodes == 0:
                    num_samples_in_capture = num_embeddings_captured // num_nodes
                    logging.info(f"Embeddings correspond to {num_samples_in_capture} samples in the first test batch.")
                    # Create metadata (ticker labels for each node embedding)
                    metadata_labels = []
                    for _ in range(num_samples_in_capture):
                        metadata_labels.extend(ticker_order)

                    if len(metadata_labels) == num_embeddings_captured:
                        try:
                            writer.add_embedding(
                                final_gnn_embeddings,
                                metadata=metadata_labels,
                                global_step=last_epoch_final,
                                tag="GNN_Output_Embeddings_Test_Final"
                            )
                            logging.info(f"Logged {num_embeddings_captured} embeddings with {len(metadata_labels)} labels to TensorBoard Projector for final run.")
                        except Exception as e:
                            logging.error(f"Error logging final embeddings to TensorBoard: {e}", exc_info=True)
                    else:
                        logging.error(f"Metadata label count ({len(metadata_labels)}) mismatch with embedding count ({num_embeddings_captured}) in final run.")
                else:
                    logging.warning(f"Embeddings count captured ({num_embeddings_captured}) is not a multiple of the number of nodes ({num_nodes}) in final run. Cannot reliably assign labels.")
            else:
                 logging.warning(f"Cannot log final embeddings due to invalid conditions (Embeddings: {num_embeddings_captured}, Nodes: {num_nodes}, Tickers: {len(ticker_order)}).")
        elif writer and hook_handle is not None: # Only warn if hook was expected
            logging.warning("No GNN embeddings captured during final evaluation, although hook was registered.")
        elif writer:
             pass # No warning needed if no hook was registered (e.g., gnn_layers=0)
        # --- END Log Embeddings ---

        # --- Log Prediction Plots to TensorBoard ---
        if writer and isinstance(test_preds_orig, np.ndarray) and isinstance(test_targets_orig, np.ndarray) and test_preds_orig.size > 0 and test_targets_orig.size > 0 and num_nodes > 0 and len(ticker_order) == num_nodes:
            logging.info("Generating final prediction plots for TensorBoard...")
            num_test_samples_processed = test_preds_orig.shape[0]

            if num_test_samples_processed > 0 and test_preds_orig.shape == test_targets_orig.shape and test_preds_orig.shape[1] == num_nodes:
                 try:
                      max_plots = 20
                      plot_indices = range(min(num_nodes, max_plots))
                      if num_nodes > max_plots: logging.warning(f"Plotting only the first {max_plots} tickers.")

                      for i in plot_indices:
                          ticker = ticker_order[i]
                          fig, ax = plt.subplots(figsize=(12, 6))
                          valid_idx = ~np.isnan(test_targets_orig[:, i]) & ~np.isnan(test_preds_orig[:, i])
                          time_axis = np.arange(num_test_samples_processed)

                          # Only plot if there are valid points for this ticker
                          if np.sum(valid_idx) > 0:
                              ax.plot(time_axis[valid_idx], test_targets_orig[valid_idx, i], label='Actual Close', color='blue', alpha=0.7, marker='.', markersize=4, linestyle='')
                              ax.plot(time_axis[valid_idx], test_preds_orig[valid_idx, i], label='Predicted Close', color='red', alpha=0.7, marker='x', markersize=4, linestyle='')
                              ax.set_title(f'Final Test Set Predictions vs Actuals - {ticker}')
                              ax.set_xlabel('Test Sequence Index')
                              ax.set_ylabel('Price (Original Scale)')
                              ax.legend()
                              ax.grid(True)
                              writer.add_figure(f'FinalRun/Predictions_vs_Actuals/Ticker_{ticker}', fig, global_step=last_epoch_final)
                          else:
                              logging.warning(f"Skipping plot for ticker {ticker} due to no valid (non-NaN) data points.")
                          plt.close(fig) # Close figure regardless of plotting
                      logging.info(f"Finished generating final prediction plots for {len(plot_indices)} tickers.")

                 except IndexError as e:
                      logging.error(f"IndexError during final plotting. Preds shape: {test_preds_orig.shape}, Targets shape: {test_targets_orig.shape}, Num nodes: {num_nodes}. Error: {e}", exc_info=True)
                 except Exception as e:
                      logging.error(f"Error during final plot generation: {e}", exc_info=True)
            else:
                 logging.warning(f"Final Prediction/Target arrays invalid or shape mismatch for plotting. Preds shape: {test_preds_orig.shape}, Targets shape: {test_targets_orig.shape}, Expected nodes: {num_nodes}.")
        elif writer:
            logging.warning("Final test predictions/targets not available/valid for plotting.")
        # --- END Log Prediction Plots ---


        # --- Log Hyperparameters and Final Metrics ---
        if writer:
             hparam_dict_final = best_params.copy() # Start with Optuna's best params
             # Add fixed parameters or other relevant info
             hparam_dict_final['sequence_length'] = SEQUENCE_LENGTH
             hparam_dict_final['prediction_horizon'] = PREDICTION_HORIZON
             hparam_dict_final['num_tickers'] = num_nodes
             hparam_dict_final['final_epochs_run'] = last_epoch_final
             # Ensure gnn_heads is present even if gnn_layers was 0
             if 'gnn_heads' not in hparam_dict_final:
                 hparam_dict_final['gnn_heads'] = 1 # Or whatever default was used

             # Combine with final metrics
             metrics_dict_final = final_metrics_dict

             # Clean metric values (replace NaN/Inf with placeholder if needed by add_hparams)
             metrics_dict_final_clean = {}
             for k, v in metrics_dict_final.items():
                 try:
                     metric_val = float(v)
                     if np.isnan(metric_val) or np.isinf(metric_val):
                          metrics_dict_final_clean[k] = -999.0 # Placeholder
                     else:
                          metrics_dict_final_clean[k] = metric_val
                 except (ValueError, TypeError):
                     metrics_dict_final_clean[k] = -999.0 # Placeholder


             try:
                 # Use '.' as run_name to log HParams within the current run directory
                 writer.add_hparams(hparam_dict_final, metrics_dict_final_clean, run_name='.')
                 logging.info("Logged final hyperparameters and test metrics to TensorBoard.")
             except Exception as e:
                 # Catch specific error related to protobuf size limit if it occurs
                 if "less than upper bound" in str(e):
                      logging.warning(f"Could not log final hyperparameters to TensorBoard due to size limit: {e}. Consider reducing the number of hparams or metrics.")
                 else:
                      logging.warning(f"Could not log final hyperparameters to TensorBoard: {e}", exc_info=True)

    except Exception as e:
         logging.error(f"An error occurred during the final run: {e}", exc_info=True)
    finally:
        # --- Cleanup ---
        if hook_handle:
            try:
                hook_handle.remove()
                logging.info("Removed forward hook from final model.")
            except Exception as e:
                logging.error(f"Error removing final hook: {e}", exc_info=True)

        if writer: # Check if writer was successfully created
            writer.close()
            logging.info("Final TensorBoard writer closed.")

        logging.info("--- Script Finished ---")

# --- END OF FILE GNN_forecast_tuned.py ---