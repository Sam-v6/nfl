# Base
import os
import random
import time
import pickle
import logging

# Common
import pandas as pd
import numpy as np

# Type hinting

# Sklearn utils
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
import joblib

# PyTorch
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

# Local
from common.decorators import timeit

@timeit
def filter_plays(plays_df: pd.DataFrame) -> pd.DataFrame:
    """Filter plays to include only relevant passing plays for coverage analysis."""

    # Create a copy
    filtered_plays_df = plays_df.copy()

    # Find the starting plays
    logging.info("Filtering data...")
    original_play_length = len(filtered_plays_df)
    logging.info(f'Total plays: {original_play_length}')

    # Filter out penalties
    filtered_plays_df = filtered_plays_df[filtered_plays_df['playNullifiedByPenalty'] == 'N']
    # Filter out rows with 'PENALTY' in the 'playDescription' column
    filtered_plays_df = filtered_plays_df[~filtered_plays_df['playDescription'].str.contains("PENALTY", na=False)]
    logging.info(f'Total plays after filtering out penalties: {len(filtered_plays_df)}')

    # Filter down to valid Man or Zone defensive play calls
    filtered_plays_df = filtered_plays_df[filtered_plays_df['pff_manZone'].isin(['Man', 'Zone'])]
    logging.info(f'Total plays after filtering to valid Man or Zone classifications: {len(filtered_plays_df)}')

    # Filter for only rows that indicate a pass play
    filtered_plays_df = filtered_plays_df[filtered_plays_df['passResult'].notna()]
    logging.info(f'Total plays after filtering to only pass plays: {len(filtered_plays_df)}')

    # Filter for only plays where the win probablity isn't lopsided (between 0.2 and 0.8)
    filtered_plays_df = filtered_plays_df[(filtered_plays_df['preSnapHomeTeamWinProbability'] > 0.1) & (filtered_plays_df['preSnapHomeTeamWinProbability'] < 0.9)]
    logging.info(f'Total plays after filtering out garbage time: {len(filtered_plays_df)}')

    # Filter for only third down or fourth down plays
    # filtered_plays_df = filtered_plays_df[filtered_plays_df['down'].isin([3, 4])]
    # logging.info(f'Total plays after filtering for 3rd or 4th down: {len(filtered_plays_df)}')

    # Filter for plays that are in our gameIds (in location data df)
    filtered_plays_df = filtered_plays_df[filtered_plays_df['gameId'].isin(location_data_df['gameId'].unique())]
    logging.info(f'Total plays after making sure they are in our location data: {len(filtered_plays_df)}')

    # Log final columns
    logging.info(filtered_plays_df.columns)

    # Cut down to columns we care about
    keep_cols_from_plays = ['gameId', 'playId', 'possessionTeam', 'defensiveTeam', 'pff_manZone']
    filtered_plays_df = filtered_plays_df.loc[:, keep_cols_from_plays].drop_duplicates()

    # Make sure we don't have any NAs in this cut down col df
    filtered_plays_df.dropna()
    logging.info(f'Total plays after cutting down to our cols and dropping NAs: {len(filtered_plays_df)}')

    # Return
    return filtered_plays_df

@timeit
def create_merged_df(location_data_df: pd.DataFrame, filtered_plays_df: pd.DataFrame) -> pd.DataFrame:
    """Merge location data with filtered plays data to create a comprehensive dataset for coverage analysis."""

    logging.info("Merging location data with filtered plays data...")

    # Create a copy of the location tracking data, cut it down to columns we care about
    loc_trimmed_df = location_data_df.copy()
    keep_cols  = [
        'gameId',
        'playId',
        'nflId',
        'frameId',
        'frameType',
        'club',
        'x',
        'y',
        's',
        'a'
    ]
    loc_trimmed_df = location_data_df.loc[:, keep_cols]

    # Cut down location tracking data copy to only before the snap and where the team isn't valid
    loc_trimmed_df = loc_trimmed_df[(loc_trimmed_df["frameType"] == "BEFORE_SNAP") & (loc_trimmed_df["club"] != "football")]

    # See the merged df that has gameId, playId, frameID all before SNAP, with x, y, and offense/defense
    logging.info(loc_trimmed_df.head())

    # Merge the two datasets such that we can have the possession and defensive team for each row
    merged_df = pd.merge(filtered_plays_df, loc_trimmed_df, on=['gameId', 'playId'], how='inner')

    # Tag the "side" of the player for each row (that being "off" or "def")
    merged_df['side'] = np.where(merged_df['club'] == merged_df['possessionTeam'], 'off', 'def')

    # Drop some columns we don't need anymore
    merged_df = merged_df.drop(['possessionTeam', 'defensiveTeam', 'club', 'frameType'], axis=1)

    # Sort for deterministic frame ordering
    merged_df = merged_df.sort_values(['gameId','playId','frameId'])

    # Let's see what we have
    logging.info(merged_df.head())

    return merged_df

def _determine_sequence_length(merged_df: pd.DataFrame) -> int:
    """Determine the maximum sequence length for the dataset based on frames per play."""

    frame_counts = (merged_df
                .groupby(['gameId','playId'])['frameId']
                .nunique())
    min_frames = int(np.percentile(frame_counts.values, 10))
    logging.info(f"Using plays that have above {min_frames} frames")

    return min_frames

def _exactly_eleven_per_side(play_df: pd.DataFrame) -> bool:
    return (
        play_df.loc[play_df.side == 'off', 'nflId'].nunique() == 11 and
        play_df.loc[play_df.side == 'def', 'nflId'].nunique() == 11
    )

def _slot_order_by_left_to_right(play_df: pd.DataFrame, side: str) -> list:
    side_df = play_df.loc[play_df["side"] == side]
    stats = (side_df
             .groupby('nflId', as_index=True)[['x','y']]
             .median()
             .rename(columns={'x':'x_med','y':'y_med'})
             .sort_values(['x_med','y_med']))
    return stats.index.tolist()  # list of sorted NFL player ids for this play to determine median x --> y player locs

def _build_side_feature_cube(play_df: pd.DataFrame, side: str, frames: np.ndarray, feature_cols: tuple) -> np.ndarray:
    """
    For one side ('off' or 'def'), build a 3D tensor:
      (T, 11, F) where F=len(feature_cols), with rows aligned to `frames`
      and slots 0..10 as columns. Missing -> NaN.
    """

    # pivot to (frames x slots) for x and y, fill missing with NaN, then stack → (min_frames, 11, F)
    side_df = play_df.loc[play_df["side"] == side]

    mats = []
    for col in feature_cols:
        # Takes the long df and goes from frameId, slot, x, y as cols to:
        # slot, 0, 1, 2 as cols ... with frameId 1, frameId 2... etc as the rows....shape is (min_frames, 11)
        mat = side_df.pivot_table(index="frameId", columns="slot", values=col)
        # It's possible certain players don't have exact tracking data throughout (ie one player has frame 10 and 12 but not frame 11), this will end up breaking our shape and cause issues downstream for model training
        # So this forces the matrix to have for each frame 
        mat = mat.reindex(index=frames, columns=range(11), fill_value=np.nan)
        mats.append(mat.to_numpy())  # shape: (T, 11)

    # stack features on the last axis to shape: (T, 11, F)
    return np.stack(mats, axis=-1)

@timeit
def build_frame_data(merged_df: pd.DataFrame) -> tuple[dict, dict]:
    """Build frame data cubes for offense and defense from the merged dataframe."""

    min_frames = _determine_sequence_length(merged_df)

    # Init series maps
    off_series = {}
    def_series = {}

    # Lists to peek at later if we skip plays
    skipped_wrong_player_count_list = []   # plays where offense or defense had >11 unique players
    skipped_under_min_frames_list = []     # plays with fewer than min_frames

    # Iterate on each play
    for (game_id, play_id), play in merged_df.groupby(['gameId','playId'], sort=False):
        # Skip if not 11 players
        if not _exactly_eleven_per_side(play):
            skipped_wrong_player_count_list.append((game_id, play_id))
            continue

        # Define slot maps (left→right by median x, tie-break median y)
        off_slots = _slot_order_by_left_to_right(play, 'off')
        def_slots = _slot_order_by_left_to_right(play, 'def')

        # Create a map that goes player id --> index so we can assign each player to an index as we go frame by frame
        off_id2slot = {pid: i for i, pid in enumerate(off_slots)}
        def_id2slot = {pid: i for i, pid in enumerate(def_slots)}

        # Assign slots (if offense use offensive map, if defense, use defensive map)
        tmp = play.copy()
        tmp['slot'] = np.where(
            tmp['side'] == 'off',
            tmp['nflId'].map(off_id2slot),
            tmp['nflId'].map(def_id2slot)
        )

        # Choose frame window (last min_frames frames)
        frames_all = np.sort(tmp['frameId'].unique())
        if frames_all.size < min_frames:
            skipped_under_min_frames_list.append((game_id, play_id))
            continue
        frames = frames_all[-min_frames:]  # Get the last min frames, so each play is consistent

        # Build offense/defense cubes: (min_frames, 11, 2) the 2 is x and y coords
        feature_cols = ("x", "y", "s", "a")
        off_arr = _build_side_feature_cube(tmp, "off", frames, feature_cols)
        def_arr = _build_side_feature_cube(tmp, "def", frames, feature_cols)

        off_series[(game_id, play_id)] = off_arr
        def_series[(game_id, play_id)] = def_arr

    logging.info(f"Kept plays: {len(off_series)}")
    logging.info(f"Skipped (>11 players): {len(skipped_wrong_player_count_list)}")
    logging.info(f"Skipped (<{min_frames} frames): {len(skipped_under_min_frames_list)}")


    return off_series, def_series

def _impute_timewise(X_np: np.ndarray) -> np.ndarray:
    """
    X_np: (T, F) with NaNs.
    Impute per feature (column) along time:
      1) forward-fill if we miss a frame, assume the player stayed where he was last seen.
      2) back-fill
      3) fill remaining NaNs with column mean (0 if all NaN)
    """
    df = pd.DataFrame(X_np)                 # (T, 11 players * 2 features = 44)

    # Copy the last known values fwd in time if there's missing NaNs
    # Fill any leading NaNs that had no earlier data
    # Example: [NaN, 3, 4, NaN, NaN, 7] ---> foward fill [NaN, 3, 4, 4, 4, 7] ---> backward fill [3, 3, 4, 4, 4, 7]
    # If a whole column has NaNs we then fill it with 0s (only time this realistically kicks in)
    df = df.ffill().bfill().fillna(0.0)
    
    return df.values.astype(np.float32)

@timeit
def build_plays_data_numpy(off_series, def_series):

    # Build labels dict mapping of (gameId, playId) --> 0/1
    label_map = {'Man': 1, 'Zone': 0}
    labels_dict = {(r.gameId, r.playId): label_map[r.pff_manZone] for r in filtered_plays_df.itertuples()}

    X_np, y_np = [], []
    for key, off_arr in off_series.items():
        def_arr = def_series[key]
        X_play = np.concatenate([off_arr, def_arr], axis=1).reshape(off_arr.shape[0], -1)  # (T, 22 * F)
        X_play = _impute_timewise(X_play)
        X_np.append(X_play.astype(np.float32))
        y_np.append(labels_dict[key])
    return X_np, np.array(y_np, dtype=int)

@timeit
def create_dataloaders(X_np: np.ndarray, y_np: np.ndarray) -> tuple[DataLoader, DataLoader, np.ndarray]:

    # Splittys
    idx_train, idx_val = train_test_split(
        np.arange(len(X_np)),               # Create an array from 0 to x number of plays
        test_size=0.2,                      # Choosing standard 20% for test size
        random_state=42,                    # Life universe and everything
        stratify=y_np                       # Says split the data while keeping same ratio of 0s and 1s in both train and validation sets
        )

    # Combine all frames from training plays
    train_stacked = np.vstack([X_np[i] for i in idx_train])  # shape: (total_train_frames, 22*F)

    # Scale data
    scaler = StandardScaler()
    scaler.fit(train_stacked)  # computes mean_ and scale_ only on training data
    joblib.dump(scaler, "plays_standard_scaler.pkl")

    def apply_scaler_to_list(X_list, idxs, scaler):
        for i in idxs:
            X_list[i] = scaler.transform(X_list[i])

    apply_scaler_to_list(X_np, idx_train, scaler)
    apply_scaler_to_list(X_np, idx_val, scaler)

    # Make tensor datasets
    # NOTE: X will be of shape (play_count, min_frames, 44)
    # NOTE: Y will be of shape (play_count, )
    train_ds = TensorDataset(
        torch.stack([torch.from_numpy(X_np[i]).float() for i in idx_train]), # Each x is (min_frame, 22*F)
        torch.from_numpy(y_np[idx_train]).long()
    )
    val_ds = TensorDataset(
        torch.stack([torch.from_numpy(X_np[i]).float() for i in idx_val]),
        torch.from_numpy(y_np[idx_val]).long()
    )

    # Reproducibility seeds
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # For deterministic behavior (slower, optional)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Make dataloaders
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)

    return train_loader, val_loader, idx_train

class LSTMClassifier(nn.Module):

    def __init__(self, input_size=44, hidden_size=64, num_layers=1, dropout=0.0, bidir=False, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidir,
        )
        out_dim = hidden_size * (2 if bidir else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, num_classes)
        )

    def forward(self, x):  # x: (B, T, F)
        out, (h_n, c_n) = self.lstm(x)        # out: (B, T, H)
        last = out[:, -1, :]                  # use last timestep representation
        logits = self.head(last)              # (B, C)
        return logits

@timeit
def train_model(train_loader: DataLoader, val_loader: DataLoader, y_np: np.ndarray, idx_train: np.ndarray) -> LSTMClassifier:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMClassifier(input_size=88, hidden_size=64, num_layers=3, dropout=0.0, bidir=False).to(device)

    # Create criterion with CE losss weighted with class weights to account for higher proportion of man coverage
    # Zone dominates class weighting, calc distribution then assign man a higher waiting on the CE loss
    y_train = y_np[idx_train]                                                      # Slice to the training fold
    classes = np.array([0, 1], dtype=int)                                           # 0=Zone, 1=Man
    w = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    logging.info("Class weights (Zone, Man):", w)
    class_weights = torch.tensor(w, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Using Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train
    for epoch in range(200):
        # Train
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Val
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                pred = model(X).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.numel()
        logging.info(f"Epoch {epoch+1}: val acc = {correct/total:.3f}")

        return model

@timeit
def viz_results(val_loader: DataLoader, model: LSTMClassifier) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    all_preds, all_true = [], []

    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X).argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_true.extend(y.cpu().numpy())

    logging.info(classification_report(all_true, all_preds, target_names=["Zone", "Man"]))

    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt
    cm = confusion_matrix(all_true, all_preds, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Zone", "Man"])
    disp.plot(cmap='Blues', values_format='d')
    plt.title("Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    
    # Configure basic logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Get raw data
    from common.data_loader import RawDataLoader
    loader = RawDataLoader()
    games_df, plays_df, players_df, location_data_df = loader.get_data(weeks=[week for week in range (1,10)])

    # Filter data
    filtered_plays_df = filter_plays(plays_df)

    # Create merged df
    merged_df = create_merged_df(location_data_df, filtered_plays_df)

    # Create cube data
    off_series, def_series = build_frame_data(merged_df)

    # Impute data and convert to numpy
    X_np, y_np = build_plays_data_numpy(off_series, def_series)

    # Create dataloaders
    train_loader, val_loader, idx_train = create_dataloaders(X_np, y_np)

    # Train model
    model = train_model(train_loader, val_loader, y_np, idx_train)

    # Create classification report and viz confusion matrix
    viz_results(val_loader, model)