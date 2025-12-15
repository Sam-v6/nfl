#!/usr/bin/env python
"""
Processes predictions to create various plots and animations to interpret model results.
"""

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

from common.paths import PROJECT_ROOT
from load_data import RawDataLoader


def plot_accuracy_across_frames(s: int, w: int, df: pd.DataFrame) -> None:
	"""
	Plots accuracy pre snap and a few moments post SNAP for all provided plays in a week.

	Inputs:
	- s: Season number, used to label plot title.
	- w: Week number, , used to label plot title.
	- df: Dataframe of predictions for all plays in a specific week.

	Outputs:
	- Plot saved to data/inference/*.png.
	"""

	# Create a new human readable column
	df["seconds_from_snap"] = df["frames_from_snap"] / 10

	# Group all the rows of the same time step together then take the mean of those rows (1 or 0) to yield accuracy
	accuracy_by_frame = df.groupby("seconds_from_snap")["base_correct"].mean()

	# Colors (NGS-ish palette)
	BG = "#0b1736"  # dark navy
	AX_BG = "#121e45"  # panel navy
	GRID = "#3b4268"  # grid gray-blue
	LINE = "#00e28e"  # neon green
	ACCENT = "#39c0ff"  # cyan for highlights
	TEXT = "#dfe6ff"  # off-white text
	SUBT = "#9aa5d1"  # subdued text

	fig, ax = plt.subplots(figsize=(10, 6), dpi=120, facecolor=BG)
	ax.set_facecolor(AX_BG)

	# Plot
	ax.plot(accuracy_by_frame.index, accuracy_by_frame.values, marker="o", linestyle="-", linewidth=2.8, markersize=5, color=LINE, label="Accuracy")

	# Grid + ticks (match animation)
	ax.yaxis.set_major_locator(MultipleLocator(0.1))
	ax.xaxis.set_minor_locator(AutoMinorLocator(4))
	ax.yaxis.set_minor_locator(AutoMinorLocator(2))

	ax.grid(True, which="major", color=GRID, linewidth=1.0, alpha=0.6)
	ax.grid(True, which="minor", color=GRID, linewidth=0.6, alpha=0.35)

	# Vertical line at snap (t = 0)
	ax.axvline(0.0, color=ACCENT, linestyle="--", linewidth=1.2, alpha=0.9)

	# Labels & title (dark theme)
	ax.set_xlabel("Seconds from Snap", color=TEXT, fontweight="bold")
	ax.set_ylabel("Accuracy", color=TEXT, fontweight="bold")
	ax.set_title("Season {s}: Week {w} Accuracy", loc="left", pad=8, color=TEXT, fontsize=10)

	# Tick colors
	ax.yaxis.set_major_locator(MultipleLocator(0.02))  # tick every 0.02
	ax.tick_params(axis="x", colors=TEXT)
	ax.tick_params(axis="y", colors=TEXT)

	# Legend styling
	legend = ax.legend(facecolor=AX_BG, edgecolor=GRID)
	for text in legend.get_texts():
		text.set_color(SUBT)

	fig.tight_layout()
	plt.savefig(PROJECT_ROOT / "data" / "inference" / f"plot_s{s}_w{w}_accuracy.png", dpi=300)


def animate_play(df: pd.DataFrame, game_id: int, play_id: int, quarter: int, play_description: str, actual_coverage: str, specific_coverage: str) -> None:
	"""
	Animates a specific play to show accuracy across frames, mostly before snap with a few frames shown post snap.

	Inputs:
	- df: Dataframe of predictions for all plays in a specific week.
	- game_id: game id to lookup in df.
	- play_id: play id to lookup in df.
	- quarter: quarter that we are looking at for plot.
	- play_description: provided play description.
	- actual_coverage: Man or Zone designation.
	- specific_coverage: Precise coverage (i.e. Cover-1)

	Outputs:
	- Animated plot saved to data/inference/*.gif
	"""

	# Create the unique id for lookup in our df
	uuid = f"{game_id}_{play_id}"

	# Create the plot df
	plot_df = df[(df["uniqueId"] == uuid) & (df["frames_from_snap"] < 300) & (df["frames_from_snap"] > -200)].copy()
	plot_df["seconds_from_snap"] = plot_df["frames_from_snap"] / 10  # Create human readable seconds (frames are at 10 Hz)
	plot_df = plot_df.sort_values("seconds_from_snap")  # Sort for seconds from snap
	plot_df = plot_df.dropna(subset=["man_prob"])  # Drop rows with missing y to avoid NaN issues
	plot_df = plot_df.iloc[::10, :].reset_index(drop=True)  # Downsample (since frame logging at 10 Hz, logging at 1 Hz)
	if plot_df.empty:
		print(f"[skip] No frames for {uuid}")
		return

	# Push to numpy
	x_all = plot_df["seconds_from_snap"].to_numpy()
	y_all = plot_df["man_prob"].to_numpy()

	# Colors (NGS-ish palette)
	BG = "#0b1736"  # dark navy
	AX_BG = "#121e45"  # panel navy
	GRID = "#3b4268"  # grid gray-blue
	LINE = "#00e28e"  # neon green
	ACCENT = "#39c0ff"  # cyan for highlights
	TEXT = "#dfe6ff"  # off-white text
	SUBT = "#9aa5d1"  # subdued text

	# Create plot
	fig, ax = plt.subplots(figsize=(10, 6), dpi=120, facecolor=BG)
	ax.set_facecolor(AX_BG)
	(line,) = ax.plot([], [], lw=2.8, color=LINE, label="Man Probability")
	dot = ax.plot([], [], marker="o", markersize=6, color=ACCENT, lw=0)[0]
	txt = ax.text(0.99, 0.98, "", transform=ax.transAxes, ha="right", va="top", color=SUBT, fontsize=10)
	ax.set_xlim(x_all.min(), x_all.max())
	ax.set_ylim(0, 1)
	ax.yaxis.set_major_locator(MultipleLocator(0.1))
	ax.grid(True, which="major", color=GRID, linewidth=1.0, alpha=0.6)
	ax.grid(True, which="minor", color=GRID, linewidth=0.6, alpha=0.35)
	ax.xaxis.set_minor_locator(AutoMinorLocator(4))
	ax.yaxis.set_minor_locator(AutoMinorLocator(2))
	# Create title text
	coverage = "Man" if actual_coverage == 1 else "Zone"
	title_text = f"Q{quarter}: {play_description}\n Actual Coverage: {coverage} - {specific_coverage}\n"

	# Continued formatting
	ax.set_title(title_text, loc="left", pad=8, color=TEXT, fontsize=10)
	ax.set_xlabel("Seconds from Snap", color=TEXT, fontweight="bold")
	ax.set_ylabel("Man Probability", color=TEXT, fontweight="bold")
	ax.axvline(0.0, color=ACCENT, linestyle="--", linewidth=1.2, alpha=0.9)

	# Make tick labels match dark theme
	ax.tick_params(axis="x", colors=TEXT)
	ax.tick_params(axis="y", colors=TEXT)

	##########################################################
	# Animation
	##########################################################
	def init() -> None:
		line.set_data([], [])
		dot.set_data([], [])
		txt.set_text("")
		return (line, dot, txt)

	def update(i: int) -> None:
		# i goes from 0..len(x_all)-1
		x = x_all[: i + 1]  # include current frame
		y = y_all[: i + 1]
		line.set_data(x, y)
		# IMPORTANT: wrap scalars as sequences
		dot.set_data([x[-1]], [y[-1]])
		txt.set_text(f"t = {x[-1]:.2f}s   P(man) = {y[-1]:.2f}")
		return (line, dot, txt)

	ani = animation.FuncAnimation(fig, update, frames=len(x_all), init_func=init, blit=True, interval=1, repeat=False)
	fig.tight_layout()
	ani.save(PROJECT_ROOT / "data" / "inference" / f"play_{uuid}.gif", writer="pillow", fps=60)
	plt.close(fig)


def get_top_man_coverage_prob_increase_plays(predictions_df: pd.Dataframe) -> pd.Dataframe:
	"""
	Grab plays that have the largest increase in man coverage probability, end with over 80% man prob and are passes.

	Inputs:
	- df: Dataframe of predictions for all plays in a specific week.

	Outputs:
	- Datafrane of top 50 plays
	"""
	# Get a df that is only pre snap and pass attempts, also that is sorted by frames from snap
	pre_snap_df = predictions_df[(predictions_df["frames_from_snap"] < 0) & (predictions_df["passAttempt"] == 1)].copy()
	pre_snap_df = pre_snap_df.sort_values(["gameId", "playId", "frames_from_snap"])

	# For each play, get the min (earliest frame) and max (latest pre-snap frame) man_prob
	change_df = (
		pre_snap_df.groupby(["gameId", "playId"])
		.agg(
			man_prob_start=("man_prob", "first"),  # Assign first value of man probability to man_prob_start
			man_prob_end=("man_prob", "last"),  # Assign last value of man probability to man_prob end
		)
		.reset_index()
	)

	# Compute the largest change and create a view that is this sorted
	change_df["man_prob_change"] = change_df["man_prob_end"] - change_df["man_prob_start"]
	largest_man_prob_increase = change_df.sort_values("man_prob_change", ascending=False)
	largest_man_prob_increase = largest_man_prob_increase[largest_man_prob_increase["man_prob_end"] > 0.8]
	top_50_plays = largest_man_prob_increase.head(50)
	return top_50_plays


def main() -> None:
	"""
	Driver to process predictions and make plots.

	Inputs:
	- None.

	Outputs:
	- None.
	"""

	# Load predictions
	predictions = []
	for s in [2022]:
		for w in [9]:
			preds_week = pd.read_parquet(PROJECT_ROOT / "data" / "inference" / f"tracking_s{s}_w{w}_preds.parquet")
			predictions.append(preds_week)
			print(f"Finished processing seasons {s}, week {w}...")
	predictions_df = pd.concat(predictions, ignore_index=True)
	predictions_df["base_correct"] = predictions_df["pred"] == predictions_df["actual"]

	# Take 15s before snap and 1.5s post snap
	week_truncated_play_df = predictions_df[(predictions_df["week"] == 9) & (predictions_df["frames_from_snap"] < 15) & (predictions_df["frames_from_snap"] > -150)].copy()

	# Create plot
	plot_accuracy_across_frames(s=s, w=w, df=week_truncated_play_df)

	# Get top x plays that have the large man coverage prob increase and are pass plays
	top_plays_df = get_top_man_coverage_prob_increase_plays

	# Load in plays df
	rawLoader = RawDataLoader()
	games_df, plays_df, players_df = rawLoader.get_base_data()

	# Merge
	merged_df = predictions_df.merge(plays_df, on=["gameId", "playId"], how="left")

	# Create animations
	for _, row in top_plays_df.iterrows():
		play = merged_df[(merged_df["playId"] == row["playId"]) & (merged_df["gameId"] == row["gameId"])]
		if play["passResult"].values[0] == "C":
			print(f"gameId: {play['gameId'].iloc[0]}, playId: {play['playId'].iloc[0]}, Q: {play['quarter'].iloc[0]}, {play['playDescription'].iloc[0]}")
			animate_play(
				df=week_truncated_play_df,
				game_id=play["gameId"].iloc[0],
				play_id=play["playId"].iloc[0],
				quarter=play["quarter"].iloc[0],
				play_description=play["playDescription"].iloc[0],
				actual_coverage=play["actual"].iloc[0],
				specific_coverage=play["pff_passCoverage"].iloc[0],
			)


if __name__ == "__main__":
	main()
