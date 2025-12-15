#!/usr/bin/env python
"""
Runs the trained transformer to generate man/zone predictions for tracking data.
"""

import json
import logging

import numpy as np
import pandas as pd
import polars as pl
import torch
from torch.profiler import ProfilerActivity, profile, record_function

from clean_data import calculate_velocity_components, label_offense_defense_manzone, make_plays_left_to_right, pass_attempt_merging, rotate_direction_and_orientation
from common.args import parse_args
from common.decorators import set_time_decorators_enabled, time_fcn
from common.paths import PROJECT_ROOT
from load_data import RawDataLoader
from models.transformer import create_transformer_model


def process_week_data_preds(rawLoader: RawDataLoader, s: int, w: int, plays: pd.DataFrame) -> pd.DataFrame:
	"""
	Loads, cleans, and labels a single week's tracking data.

	Inputs:
	- rawLoader: Data laoder object to retrieve weekly tracking data.
	- s: Season number to process.
	- w: Week number to process.
	- plays: Play metadata with labels.

	Outputs:
	- week_df: Cleaned and labeled tracking rows for the requested week.
	"""

	# Load data
	week = rawLoader.get_tracking_data(weeks=[w], seasons=[s])

	# Apply cleaning functions
	week = rotate_direction_and_orientation(week)
	week = make_plays_left_to_right(week)
	week = calculate_velocity_components(week)
	week = pass_attempt_merging(week, plays)
	# week = label_offense_defense_coverage(week, plays)  # for specific coverage... currently set to man/zone only
	week = label_offense_defense_manzone(week, plays)

	week["week"] = w
	week["uniqueId"] = week["gameId"].astype(str) + "_" + week["playId"].astype(str)
	week["frameUniqueId"] = week["gameId"].astype(str) + "_" + week["playId"].astype(str) + "_" + week["frameId"].astype(str)

	# Add frames_from_snap (to do: make this a function but fine for now)
	snap_frames = week[week["event"] == "ball_snap"].groupby("uniqueId")["frameId"].first()
	week = week.merge(snap_frames.rename("snap_frame"), on="uniqueId", how="left")
	week["frames_from_snap"] = week["frameId"] - week["snap_frame"]

	# filtering only for even frames
	# week = week[week['frameId'] % 2 == 0]

	# Ridding of any potential outliers (25 seconds after the snap)
	week = week[(week["frames_from_snap"] >= -150) & (week["frames_from_snap"] <= 30)]

	# applying data augmentation to increase training size (centered around 0-4 seconds presnap!)
	# -- 1/3rd of the current num of frames... specifically selecting for frames around the snap

	# num_unique_frames = len(set(week['frameUniqueId']))
	# selected_frames = select_augmented_frames(week, int(num_unique_frames / 3), sigma=5)
	# week_aug = data_augmentation(week, selected_frames)

	# week = pd.concat([week, week_aug])

	logging.info(f"Finished processing seasson {s} week {w} data")

	return week


def prepare_tensor(play: pd.DataFrame) -> torch.Tensor:
	"""
	Converts a single play slice into a model-ready tensor.

	Inputs:
	- play: Tracking rows for one frameUniqueId.

	Outputs:
	- frame_tensor: Tensor shaped [frames, players, features] for inference.
	"""
	features = ["x_clean", "y_clean", "v_x", "v_y", "defense"]
	play_data = play[features + ["frameId"]]
	play_data = play_data.sort_values(by="frameId")

	frames = play_data.groupby("frameId")[features].apply(lambda g: g.to_numpy())
	all_frames_tensor = np.stack(frames.to_list())  # Shape: [num_frames, num_players, num_features]
	all_frames_tensor = torch.tensor(all_frames_tensor, dtype=torch.float32)

	return all_frames_tensor  # Shape: [num_frames, num_players, num_features]


def profile_inference(model: torch.nn.Module, device: torch.device) -> None:
	"""
	Profiles inference of a single frame using a saved example tensor.

	Inputs:
	- model: The transformer model to profile.
	- device: The device to run the model on.

	Outputs:
	- Saved trace_inference.json file in log/profiler/ to visualize in Chrome/Perfetto.
	"""

	example_path = PROJECT_ROOT / "data" / "inference" / "example_frame_tensor.pt"
	if example_path.exists():
		frame_tensor = torch.load(example_path).to(device)

		# Warmup to trigger torch.compile JIT and CUDA autotuning outside profiler
		with torch.no_grad():
			for _ in range(5):
				_ = model(frame_tensor)

		log_dir = PROJECT_ROOT / "log" / "profiler"
		log_dir.mkdir(parents=True, exist_ok=True)
		trace_path = log_dir / "trace_inference.json"

		# Setup profiler
		activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
		with (
			torch.no_grad(),
			profile(
				activities=activities,
				record_shapes=True,
				profile_memory=True,
				with_stack=True,
			) as prof,
		):
			# Run several inferences to get stable averages
			for _ in range(20):
				with record_function("inference_step"):
					_ = model(frame_tensor)

		print("Top CPU ops (inference):")
		print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))
		print("Top CUDA ops (inference):")
		print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))

		# Save Chrome/Perfetto trace
		prof.export_chrome_trace(str(trace_path))
		print(f"Inference trace saved to: {trace_path}")

		# Early return so we don't also run the full weekly loop when profiling
		return
	else:
		logging.warning("args.profile is set, but example_frame_tensor.pt not found. Run once without profiling to generate it.")


@time_fcn
def main() -> None:
	"""
	Streams weekly tracking frames through the transformer and writes predictions.

	Inputs:
	- CLI flags from parse_args control profiling.

	Outputs:
	- CSV files with frame-level predictions and merged tracking data.
	"""
	# Set logging
	logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

	# Get input args
	args = parse_args()

	# Enable/disable timing decorators
	if args.profile:
		set_time_decorators_enabled(True)
		logging.info("Timing decorators enabled")
	else:
		set_time_decorators_enabled(False)
		logging.info("Timing decorators disabled")

	# Set TensorFloat-32 (TF32) mode for matmul and cudnn (speeds up training on Ampere+ GPUs with minimal impact on accuracy)
	torch.backends.cuda.matmul.allow_tf32 = True
	torch.backends.cudnn.allow_tf32 = True

	# Set device and profiler
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	activities = [ProfilerActivity.CPU]
	activities += [ProfilerActivity.CUDA]

	# Reload model which includes:
	#    1. Load model weights that were generated as best trial from Ray HPO
	#    2. Create the model with those hyperparameters
	#    3. Load it on the GPU
	#    4. Compile the model with torch.compile for faster inference
	#    5. Load the last checkpoint of the model weights
	#    6. Set to eval for inference
	with open(PROJECT_ROOT / "data" / "training" / "model_params.json") as file:
		config = json.load(file)
	model = create_transformer_model(config)
	model = model.to(device)
	model = torch.compile(
		model,
		mode="default",  # default, reduces overhead, generally stable
		fullgraph=True,  # enable full graph fusion, helps reduce kernel launch overhead
	)
	model.load_state_dict(torch.load(PROJECT_ROOT / "data" / "training" / "transformer.pt", map_location=device))
	model.eval()

	# Load data
	rawLoader = RawDataLoader()
	games_df, plays_df, players_df = rawLoader.get_base_data()

	# Process + predict one week at a time (keeps RAM low)
	seasons_weeks = {2022: [9]}
	for s in seasons_weeks.keys():
		for w in seasons_weeks[s]:
			week_df = process_week_data_preds(rawLoader, s, w, plays_df)

			# Filtering early to shrink memory
			week_df = week_df[(week_df["club"] != "football") & (week_df["passAttempt"] == 1)].copy()

			# Polars convert optional for speed
			tracking_df_polars = pl.DataFrame(week_df)

			# Stream predictions to CSV in batches
			weekly_predictions_path_csv = PROJECT_ROOT / "data" / "inference" / f"week{w}_preds.csv"
			wrote_header = False
			batch = []

			# Iterate unique frames without building a giant Python set
			list_ids = pd.unique(week_df["frameUniqueId"].values)

			logging.info(f"Starting loop for week {w}...")
			for idx, frame_id in enumerate(list_ids, start=1):
				if idx % 20000 == 0:
					logging.info(f"Processed {idx}/{len(list_ids)} frames ({100 * idx / len(list_ids):.1f}%)")

				# Grab frame rows (polars or pandas)
				if tracking_df_polars is not None:
					frame = tracking_df_polars.filter(pl.col("frameUniqueId") == frame_id).to_pandas()
				else:
					frame = week_df.loc[week_df["frameUniqueId"] == frame_id]

				# Lightweight tensor build
				frame_tensor = prepare_tensor(frame)
				if frame_tensor is None:
					continue

				# Save the first tensor for profiling later
				if idx == 1:
					torch.save(frame_tensor, PROJECT_ROOT / "data" / "inference" / "example_frame_tensor.pt")

				# Move to device and run inference
				frame_tensor = frame_tensor.to(device, non_blocking=True)
				with torch.no_grad():
					outputs = model(frame_tensor)  # [1, 2]
					probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
					zone_prob, man_prob = float(probabilities[0]), float(probabilities[1])
					pred = 0 if zone_prob > man_prob else 1
					actual = int(frame["pff_manZone"].iloc[0]) if "pff_manZone" in frame.columns and not pd.isna(frame["pff_manZone"].iloc[0]) else -1

				play_id = "_".join(frame_id.split("_")[:2])
				frame_num = int(frame_id.split("_")[-1])

				batch.append({"frameUniqueId": frame_id, "uniqueId": play_id, "frameId": frame_num, "zone_prob": zone_prob, "man_prob": man_prob, "pred": pred, "actual": actual})

				# Flush batch to CSV to keep RAM low
				if len(batch) >= config["batch_size"]:
					pd.DataFrame(batch).to_csv(weekly_predictions_path_csv, mode="a", header=not wrote_header, index=False)
					wrote_header = True
					batch.clear()

			# Flush tail
			if batch:
				pd.DataFrame(batch).to_csv(weekly_predictions_path_csv, mode="a", header=not wrote_header, index=False)
				batch.clear()

			logging.info(f"Finished week {w}... saved to week{w}_preds.csv\n")

			# Merge week_df with preds (per-week, small)
			preds_week = pd.read_csv(weekly_predictions_path_csv, usecols=["frameUniqueId", "zone_prob", "man_prob", "pred", "actual"])
			tracking_preds = week_df.merge(preds_week, on="frameUniqueId", how="left")
			tracking_preds.to_parquet(PROJECT_ROOT / "data" / "inference" / f"tracking_s{s}_w{w}_preds.parquet", index=False)

			# Free RAM before next week
			del week_df, tracking_df_polars, preds_week, tracking_preds
			import gc

			gc.collect()

	# If profiling flag is set, run profiler
	if args.profile:
		profile_inference(model, device)


if __name__ == "__main__":
	main()
