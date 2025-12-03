#!/usr/bin/env python
"""
Trains an LSTM model to classify man versus zone coverage from tracking data.
"""

import logging

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from common.models.lstm import LSTMClassifier
from sklearn.metrics import ConfusionMatrixDisplay, auc, classification_report, confusion_matrix, precision_recall_fscore_support, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset

from common.decorators import time_fcn

PlayKey = tuple[int, int]
SeriesDict = dict[PlayKey, np.ndarray]


@time_fcn
def filter_plays(plays_df: pd.DataFrame) -> pd.DataFrame:
	"""
	Filters plays down to valid pass attempts with man/zone labels.

	Inputs:
	- plays_df: Raw plays table with penalty and label info.

	Outputs:
	- filtered_plays_df: Plays limited to pass attempts with man/zone tags.
	"""

	# Create a copy
	filtered_plays_df = plays_df.copy()

	# Find the starting plays
	logging.info("Filtering data...")
	original_play_length = len(filtered_plays_df)
	logging.info(f"Total plays: {original_play_length}")

	# Filter out penalties
	filtered_plays_df = filtered_plays_df[filtered_plays_df["playNullifiedByPenalty"] == "N"]
	# Filter out rows with 'PENALTY' in the 'playDescription' column
	filtered_plays_df = filtered_plays_df[~filtered_plays_df["playDescription"].str.contains("PENALTY", na=False)]
	logging.info(f"Total plays after filtering out penalties: {len(filtered_plays_df)}")

	# Filter down to valid Man or Zone defensive play calls
	filtered_plays_df = filtered_plays_df[filtered_plays_df["pff_manZone"].isin(["Man", "Zone"])]
	logging.info(f"Total plays after filtering to valid Man or Zone classifications: {len(filtered_plays_df)}")

	# Filter for only rows that indicate a pass play
	filtered_plays_df = filtered_plays_df[filtered_plays_df["passResult"].notna()]
	logging.info(f"Total plays after filtering to only pass plays: {len(filtered_plays_df)}")

	# Filter for only plays where the win probablity isn't lopsided (between 0.1 and 0.9), likelyhood that there's more movement
	# filtered_plays_df = filtered_plays_df[(filtered_plays_df['preSnapHomeTeamWinProbability'] > 0.1) & (filtered_plays_df['preSnapHomeTeamWinProbability'] < 0.9)]
	# logging.info(f'Total plays after filtering out garbage time: {len(filtered_plays_df)}')

	# Filter for only third down or fourth down plays
	# filtered_plays_df = filtered_plays_df[filtered_plays_df['down'].isin([3, 4])]
	# logging.info(f'Total plays after filtering for 3rd or 4th down: {len(filtered_plays_df)}')

	# TODO: Fix
	# Filter for plays that are in our gameIds (in location data df)
	filtered_plays_df = filtered_plays_df[filtered_plays_df["gameId"].isin(location_data_df["gameId"].unique())]
	logging.info(f"Total plays after making sure they are in our location data: {len(filtered_plays_df)}")

	# Log final columns
	logging.info(filtered_plays_df.columns)

	# Cut down to columns we care about
	keep_cols_from_plays = ["gameId", "playId", "possessionTeam", "defensiveTeam", "pff_manZone"]
	filtered_plays_df = filtered_plays_df.loc[:, keep_cols_from_plays].drop_duplicates()

	# Make sure we don't have any NAs in this cut down col df
	filtered_plays_df.dropna()
	logging.info(f"Total plays after cutting down to our cols and dropping NAs: {len(filtered_plays_df)}")

	# Return
	return filtered_plays_df


@time_fcn
def create_merged_df(location_data_df: pd.DataFrame, filtered_plays_df: pd.DataFrame) -> pd.DataFrame:
	"""
	Merges filtered plays with presnap location rows and tags offense/defense sides.

	Inputs:
	- location_data_df: Tracking data containing positions and frame info.
	- filtered_plays_df: Plays already narrowed to labeled pass attempts.

	Outputs:
	- merged_df: Presnap tracking rows with side tags and minimal columns.
	"""

	logging.info("Merging location data with filtered plays data...")

	# Create a copy of the location tracking data, cut it down to columns we care about
	loc_trimmed_df = location_data_df.copy()
	keep_cols = ["gameId", "playId", "nflId", "frameId", "frameType", "club", "x", "y", "s", "a"]
	loc_trimmed_df = location_data_df.loc[:, keep_cols]

	# Cut down location tracking data copy to only before the snap and where the team isn't valid
	loc_trimmed_df = loc_trimmed_df[(loc_trimmed_df["frameType"] == "BEFORE_SNAP") & (loc_trimmed_df["club"] != "football")]

	# See the merged df that has gameId, playId, frameID all before SNAP, with x, y, and offense/defense
	logging.info(loc_trimmed_df.head())

	# Merge the two datasets such that we can have the possession and defensive team for each row
	merged_df = pd.merge(filtered_plays_df, loc_trimmed_df, on=["gameId", "playId"], how="inner")

	# Tag the "side" of the player for each row (that being "off" or "def")
	merged_df["side"] = np.where(merged_df["club"] == merged_df["possessionTeam"], "off", "def")

	# Drop some columns we don't need anymore
	merged_df = merged_df.drop(["possessionTeam", "defensiveTeam", "club", "frameType"], axis=1)

	# Sort for deterministic frame ordering
	merged_df = merged_df.sort_values(["gameId", "playId", "frameId"])

	# Let's see what we have
	logging.info(merged_df.head())

	return merged_df


def _determine_sequence_length(merged_df: pd.DataFrame) -> int:
	"""
	Computes a common sequence length using the lower decile of frame counts.

	Inputs:
	- merged_df: Presnap tracking rows with frame identifiers.

	Outputs:
	- min_frames: Sequence length threshold to keep plays.
	"""

	frame_counts = merged_df.groupby(["gameId", "playId"])["frameId"].nunique()
	min_frames = int(np.percentile(frame_counts.values, 10))
	logging.info(f"Using plays that have above {min_frames} frames")

	return min_frames


def _exactly_eleven_per_side(play_df: pd.DataFrame) -> bool:
	"""
	Checks whether a play has exactly eleven unique offensive and defensive players.

	Inputs:
	- play_df: Rows for a single play.

	Outputs:
	- has_eleven: True when both sides have eleven participants.
	"""
	return play_df.loc[play_df.side == "off", "nflId"].nunique() == 11 and play_df.loc[play_df.side == "def", "nflId"].nunique() == 11


def _slot_order_by_left_to_right(play_df: pd.DataFrame, side: str) -> list[int]:
	"""
	Orders players on one side by median field position to assign slots.

	Inputs:
	- play_df: Rows for a single play.
	- side: "off" or "def" to choose which group to order.

	Outputs:
	- slot_order: List of player ids sorted left to right, then low to high y.
	"""
	side_df = play_df.loc[play_df["side"] == side]
	stats = side_df.groupby("nflId", as_index=True)[["x", "y"]].median().rename(columns={"x": "x_med", "y": "y_med"}).sort_values(["x_med", "y_med"])
	return stats.index.tolist()  # list of sorted NFL player ids for this play to determine median x --> y player locs


def _build_side_feature_cube(play_df: pd.DataFrame, side: str, frames: np.ndarray, feature_cols: tuple[str, ...]) -> np.ndarray:
	"""
	Builds a (frames x players x features) cube for one side of the ball.

	Inputs:
	- play_df: Rows for a single play with slot assignments.
	- side: "off" or "def" slice to process.
	- frames: Ordered frame ids to include.
	- feature_cols: Tuple of feature column names to stack.

	Outputs:
	- side_cube: Numpy array shaped (T, 11, F) with NaNs for missing data.
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


@time_fcn
def build_frame_data(merged_df: pd.DataFrame) -> tuple[SeriesDict, SeriesDict]:
	"""
	Constructs offense and defense frame cubes for plays that meet criteria.

	Inputs:
	- merged_df: Presnap tracking rows with side assignments.

	Outputs:
	- off_series: Mapping of (gameId, playId) to offense cubes.
	- def_series: Mapping of (gameId, playId) to defense cubes.
	"""

	min_frames = _determine_sequence_length(merged_df)

	# Init series maps
	off_series = {}
	def_series = {}

	# Lists to peek at later if we skip plays
	skipped_wrong_player_count_list = []  # plays where offense or defense had >11 unique players
	skipped_under_min_frames_list = []  # plays with fewer than min_frames

	# Iterate on each play
	for (game_id, play_id), play in merged_df.groupby(["gameId", "playId"], sort=False):
		# Skip if not 11 players
		if not _exactly_eleven_per_side(play):
			skipped_wrong_player_count_list.append((game_id, play_id))
			continue

		# Define slot maps (left→right by median x, tie-break median y)
		off_slots = _slot_order_by_left_to_right(play, "off")
		def_slots = _slot_order_by_left_to_right(play, "def")

		# Create a map that goes player id --> index so we can assign each player to an index as we go frame by frame
		off_id2slot = {pid: i for i, pid in enumerate(off_slots)}
		def_id2slot = {pid: i for i, pid in enumerate(def_slots)}

		# Assign slots (if offense use offensive map, if defense, use defensive map)
		tmp = play.copy()
		tmp["slot"] = np.where(tmp["side"] == "off", tmp["nflId"].map(off_id2slot), tmp["nflId"].map(def_id2slot))

		# Choose frame window (last min_frames frames)
		frames_all = np.sort(tmp["frameId"].unique())
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
	Imputes missing frame values by carrying forward/backward and filling zeros.

	Inputs:
	- X_np: Two-dimensional array of frame features with NaNs.

	Outputs:
	- imputed: Array with timewise imputation applied.
	"""
	df = pd.DataFrame(X_np)  # (T, 11 players * 2 features = 44)

	# Copy the last known values fwd in time if there's missing NaNs
	# Fill any leading NaNs that had no earlier data
	# Example: [NaN, 3, 4, NaN, NaN, 7] ---> foward fill [NaN, 3, 4, 4, 4, 7] ---> backward fill [3, 3, 4, 4, 4, 7]
	# If a whole column has NaNs we then fill it with 0s (only time this realistically kicks in)
	df = df.ffill().bfill().fillna(0.0)

	return df.values.astype(np.float32)


@time_fcn
def build_plays_data_numpy(off_series: SeriesDict, def_series: SeriesDict) -> tuple[list[np.ndarray], np.ndarray]:
	"""
	Stacks offense/defense cubes into play-level tensors and builds labels.

	Inputs:
	- off_series: Mapping of offense cubes keyed by (gameId, playId).
	- def_series: Mapping of defense cubes keyed by (gameId, playId).

	Outputs:
	- X_np: List of per-play tensors shaped (frames, features).
	- y_np: Array of play-level man/zone labels.
	"""
	# Build labels dict mapping of (gameId, playId) --> 0/1
	label_map = {"Man": 1, "Zone": 0}
	labels_dict = {(r.gameId, r.playId): label_map[r.pff_manZone] for r in filtered_plays_df.itertuples()}

	X_np, y_np = [], []
	for key, off_arr in off_series.items():
		def_arr = def_series[key]
		X_play = np.concatenate([off_arr, def_arr], axis=1).reshape(off_arr.shape[0], -1)  # (T, 22 * F)
		X_play = _impute_timewise(X_play)
		X_np.append(X_play.astype(np.float32))
		y_np.append(labels_dict[key])
	return X_np, np.array(y_np, dtype=int)


@time_fcn
def create_dataloaders(X_np: list[np.ndarray], y_np: np.ndarray) -> tuple[DataLoader, DataLoader, np.ndarray]:
	"""
	Splits play tensors into train/validation sets and builds loaders.

	Inputs:
	- X_np: List of play tensors.
	- y_np: Array of labels aligned with X_np.

	Outputs:
	- train_loader: DataLoader for training plays.
	- val_loader: DataLoader for validation plays.
	- idx_train: Indices used for the training split.
	"""
	# Splittys
	idx_train, idx_val = train_test_split(
		np.arange(len(X_np)),  # Create an array from 0 to x number of plays
		test_size=0.2,  # Choosing standard 20% for test size
		random_state=42,  # Life universe and everything
		stratify=y_np,  # Says split the data while keeping same ratio of 0s and 1s in both train and validation sets
	)

	# Combine all frames from training plays
	train_stacked = np.vstack([X_np[i] for i in idx_train])  # shape: (total_train_frames, 22*F)

	# Scale data
	scaler = StandardScaler()
	scaler.fit(train_stacked)  # computes mean_ and scale_ only on training data
	joblib.dump(scaler, "plays_standard_scaler.pkl")

	def apply_scaler_to_list(X_list: list[np.ndarray], idxs: np.ndarray, scaler: StandardScaler) -> None:
		for i in idxs:
			X_list[i] = scaler.transform(X_list[i])

	apply_scaler_to_list(X_np, idx_train, scaler)
	apply_scaler_to_list(X_np, idx_val, scaler)

	# Make tensor datasets
	# NOTE: X will be of shape (play_count, min_frames, 44)
	# NOTE: Y will be of shape (play_count, )
	train_ds = TensorDataset(
		torch.stack([torch.from_numpy(X_np[i]).float() for i in idx_train]),  # Each x is (min_frame, 22*F)
		torch.from_numpy(y_np[idx_train]).long(),
	)
	val_ds = TensorDataset(
		torch.stack([torch.from_numpy(X_np[i]).float() for i in idx_val]),
		torch.from_numpy(y_np[idx_val]).long(),
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


def collect_probs(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
	"""
	Gathers true labels and predicted probabilities for a dataset.

	Inputs:
	- model: Trained classifier.
	- loader: DataLoader providing batches.
	- device: Target device for inference.

	Outputs:
	- y_true: Concatenated ground-truth labels.
	- y_prob: Concatenated probabilities for the positive class.
	"""

	model.eval()  # inference mode, disables dropout and batch norm updates
	y_true, y_prob = [], []  # lists to collect true labels and predicted probabilities
	with torch.no_grad():  # no need to calcualte gradients for validation
		for x, y in loader:
			x = x.to(device)  # move to GPU
			logits = model(x)  # forward pass to get raw class scores of shape [B, 2]
			prob1 = torch.softmax(logits, dim=1)[:, 1]  # turn logits into probabilties, get the man column
			y_prob.append(prob1.cpu().numpy())  # move to CPU and convert to numpy
			y_true.append(y.numpy())  # true labels on CPU as numpy
	return np.concatenate(y_true), np.concatenate(y_prob)  # flatten all batches into single arrays


def report_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, thr: float, beta: float = 1.0) -> dict[str, float]:
	"""
	Calculates precision/recall metrics at a given probability threshold.

	Inputs:
	- y_true: Ground-truth labels.
	- y_prob: Predicted probabilities.
	- thr: Threshold for classifying positive.
	- beta: Beta value for f-score if needed.

	Outputs:
	- metrics: Dictionary of threshold and class 1 precision/recall/f-score.
	"""
	# Turns probabilities into binary predictions at threshold
	y_pred = (y_prob >= thr).astype(int)

	# Get structured metrics
	(prec0, prec1), (rec0, rec1), (f10, f11), _ = precision_recall_fscore_support(y_true, y_pred, average=None, labels=[0, 1], beta=beta, zero_division=0)

	# Return threshold, man precision, man recall, and man f1 score
	return {"thr": thr, "man_prec": prec1, "man_rec": rec1, "man_f": f11}


def tune_threshold_for_precision(y_true: np.ndarray, y_prob: np.ndarray, min_recall: float | None = None, beta: float = 0.5) -> dict[str, float]:
	"""
	Searches thresholds to maximize precision on the positive class.

	Inputs:
	- y_true: Ground-truth labels.
	- y_prob: Predicted probabilities.
	- min_recall: Optional recall floor.
	- beta: Beta value for f-score filtering.

	Outputs:
	- best: Dictionary containing best threshold and its metrics.
	"""
	# Star off with unique probs and a coarse grid of thresholds
	unique_probs = np.unique(np.clip(y_prob, 1e-6, 1 - 1e-6))
	grid = np.linspace(0.05, 0.95, 37)

	# Merge and de-duplicate
	thresholds = np.unique(np.concatenate([unique_probs, grid]))

	# Find best threshold
	best = {"thr": 0.5, "man_prec": 0.0, "man_rec": 0.0, "man_f": 0.0}
	for t in thresholds:
		y_pred = (y_prob >= t).astype(int)
		(prec0, prec1), (rec0, rec1), (f10, f11), _ = precision_recall_fscore_support(y_true, y_pred, average=None, labels=[0, 1], beta=beta, zero_division=0)
		if min_recall is not None and rec1 < min_recall:
			continue
		if prec1 > best["man_prec"]:
			best = {
				"thr": float(t),
				"man_prec": float(prec1),
				"man_rec": float(rec1),
				"man_f": float(f11),
			}

	# Return the best threshold (and associated metrics)
	return best


@time_fcn
def train_model(train_loader: DataLoader, val_loader: DataLoader, y_np: np.ndarray, idx_train: np.ndarray) -> LSTMClassifier:
	"""
	Trains the LSTM classifier with early stopping based on precision.

	Inputs:
	- train_loader: Batches for training.
	- val_loader: Batches for validation.
	- y_np: All labels to compute class weights.
	- idx_train: Indices corresponding to the training split.

	Outputs:
	- model: Trained LSTMClassifier with best threshold stored.
	"""
	# Set device to GPU
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Init LSTM model
	model = LSTMClassifier(input_size=88, hidden_size=64, num_layers=2, dropout=0.4, bidir=False, num_classes=2).to(device)

	# Create criterion with CE losss weighted with class weights to account for higher proportion of man coverage
	# Zone dominates class weighting, calc distribution then assign man a higher waiting on the CE loss
	y_train = y_np[idx_train]  # Slice to the training fold
	classes = np.array([0, 1], dtype=int)  # 0=Zone, 1=Man
	w = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
	logger.info("Class weights (Zone, Man): %s", w)
	class_weights = torch.tensor(w, dtype=torch.float32, device=device)
	criterion = nn.CrossEntropyLoss(weight=class_weights)

	# Using Adam
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

	# Init values for finding best theshold
	set_min_recall = 0.25  # minimum recall floor when tuning thresholds
	best_state = None
	best_man_prec = -1.0  # init so first epoch can win
	best_thr = 0.5  # default threshold
	patience = 10  # how many epochs to wait until early stop
	bad = 0

	# Train
	for epoch in range(50):
		#####################################
		# Train
		#####################################
		model.train()
		for X, y in train_loader:
			X, y = X.to(device), y.to(device)
			logits = model(X)
			loss = criterion(logits, y)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		#####################################
		# Validation
		#####################################
		# Collect raw probabilities on validation set
		y_true_val, y_prob_val = collect_probs(model, val_loader, device)

		# Tune threshold for max Man precision with a recall floor
		tuned = tune_threshold_for_precision(y_true_val, y_prob_val, min_recall=set_min_recall, beta=0.5)

		# Report at tuned threshold
		stats_tuned = report_at_threshold(y_true_val, y_prob_val, thr=tuned["thr"], beta=0.5)
		logging.info(f"[epoch {epoch + 1}] Tuned thr={tuned['thr']:.3f} | Man P={tuned['man_prec']:.2f} R={tuned['man_rec']:.2f}")

		# Early-stop on best Man precision-at-tuned-threshold
		if tuned["man_prec"] > best_man_prec:
			best_man_prec = tuned["man_prec"]
			best_thr = tuned["thr"]
			best_state = {k: v.cpu() for k, v in model.state_dict().items()}
			bad = 0
		else:
			bad += 1
			if bad >= patience:
				logging.info(f"Early stopping at epoch {epoch + 1}")
				break

	# Restore best weights and attach tuned threshold
	if best_state is not None:
		model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
	logging.info(f"Final chosen Man threshold = {best_thr:.3f} (best Man precision={best_man_prec:.2f})")
	model.best_man_threshold = best_thr  # <-- sets the attribute used by viz

	return model


@time_fcn
def viz_results(val_loader: DataLoader, model: LSTMClassifier) -> None:
	"""
	Evaluates the trained model on validation data and plots diagnostics.

	Inputs:
	- val_loader: Validation DataLoader.
	- model: Trained LSTM model containing tuned threshold.

	Outputs:
	- Logs metrics and renders confusion matrix and ROC plots.
	"""
	# Set device
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Use tuned threshold if available
	thr = getattr(model, "best_man_threshold", 0.5)

	# Evaluate
	model.eval()
	all_true, all_prob = [], []
	with torch.no_grad():
		for X, y in val_loader:
			X = X.to(device)
			logits = model(X)  # [B, 2]
			prob_man = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
			all_prob.extend(prob_man)
			all_true.extend(y.numpy())
	all_true = np.array(all_true)
	all_prob = np.array(all_prob)
	all_pred = (all_prob >= thr).astype(int)

	# Classification report
	logging.info(f"Using threshold = {thr:.3f}")
	logging.info(classification_report(all_true, all_pred, target_names=["Zone", "Man"]))

	# Plot confusion matrix
	cm = confusion_matrix(all_true, all_pred, labels=[0, 1])
	disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Zone", "Man"])
	disp.plot(cmap="Blues", values_format="d")
	plt.title(f"Confusion Matrix (thr={thr:.3f})")
	plt.show()

	# Plot ROC curve
	fpr, tpr, thresholds = roc_curve(all_true, all_prob)  # uses probabilities (not thresholded)
	roc_auc = auc(fpr, tpr)

	plt.figure(figsize=(6, 6))
	plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
	plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--", label="Chance line")
	plt.xlabel("False Positive Rate (1 - Specificity)")
	plt.ylabel("True Positive Rate (Recall)")
	plt.title("Receiver Operating Characteristic (ROC)")
	plt.legend(loc="lower right")
	plt.grid(True, linestyle="--", alpha=0.6)
	plt.show()
	logging.info(f"ROC AUC = {roc_auc:.3f}")


@time_fcn
def main() -> None:
	"""
	Orchestrates data prep, training, and evaluation for the LSTM workflow.

	Inputs:
	- None (configuration is defined in code).

	Outputs:
	- Trains the model and produces validation diagnostics.
	"""
	# Configure basic logging
	logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
	logger = logging.getLogger(__name__)

	# Get raw data
	from common.data_loader import RawDataLoader

	loader = RawDataLoader()
	games_df, plays_df, players_df, location_data_df = loader.get_data(weeks=[week for week in range(1, 10)])

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


if __name__ == "__main__":
	main()
