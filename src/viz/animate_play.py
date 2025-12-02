#!/usr/bin/env python
"""
Animates a single play using tracking data to visualize player movement.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
from matplotlib.artist import Artist

"""
# TODO
- Add labels on home team vs away team, score, quarter, down, distance
- Make it so different positions show up in different shape and offense/defense colors
- Investigate why the vectors seems different between offense an defense
- Make the field itself look better
- Find a comparison play on youtube so I can see the real video version
- Investigate if there's a need to show speed or acceleration?
- Tag the events with color swaps of ball (pass released flash the ball, pass caught, flash the ball, etc)
- Make a driver so I can start loading all these plays (later)
"""


def animate_play(
	game_info_df: pd.DataFrame,
	play_data_single_play_df: pd.DataFrame,
	players_df: pd.DataFrame,
	location_data_single_play_df: pd.DataFrame,
) -> None:
	"""
	Renders an animation of player movement for a single play.

	Inputs:
	- game_info_df: Metadata for the current game.
	- play_data_single_play_df: Play-level details (down, distance, etc.).
	- players_df: Player metadata table.
	- location_data_single_play_df: Tracking rows for the play.

	Outputs:
	- Displays an animated matplotlib figure of the play.
	"""

	# Unpack contents
	game_id = game_info_df["gameId"]
	week = game_info_df["week"]
	home_team = game_info_df["homeTeamAbbr"]
	away_team = game_info_df["visitorTeamAbbr"]
	home_score = game_info_df["homeFinalScore"]
	away_score = game_info_df["visitorFinalScore"]
	play_id = play_data_single_play_df["playId"].values[0]
	quarter = play_data_single_play_df["quarter"].values[0]
	down = play_data_single_play_df["down"].values[0]
	yards_to_go = play_data_single_play_df["yardsToGo"].values[0]
	play_description = play_data_single_play_df["playDescription"].values[0]

	# Grab contents of the play
	frame_ids = sorted(location_data_single_play_df["frameId"].unique())

	# Find the frame where the ball is snapped
	snap_row = location_data_single_play_df[location_data_single_play_df["event"] == "ball_snap"]
	if not snap_row.empty:
		snap_frame_id = snap_row["frameId"].min()
	else:
		if len(frame_ids) == 0:
			raise ValueError(f"No frames found for game {game_id}, play {play_id}.")
		snap_frame_id = frame_ids[0]
		print(f"No 'ball_snap' event found for game {game_id}, play {play_id}. Using first frame ({snap_frame_id}) as snap.")

	# Setup plot
	fig, ax = plt.subplots(figsize=(15, 6))
	ax.set_xlim(0, 120)
	ax.set_ylim(0, 53.3)
	ax.set_facecolor("green")
	ax.set_aspect("equal")

	# Draw football field
	for x in range(10, 111, 10):
		ax.axvline(x, color="white", linewidth=1)
	ax.axhline(0, color="white")
	ax.axhline(53.3, color="white")

	player_dots = []
	arrows = []
	labels = []

	# Placeholder for relative time display
	time_text = ax.text(60, 51, "", fontsize=12, ha="center", color="white", bbox=dict(facecolor="black", alpha=0.5))

	def init() -> list[Artist]:
		"""Initializes artists for FuncAnimation."""
		return player_dots + arrows + labels + [time_text]

	def update(frame_id: int) -> list[Artist]:
		"""
		Updates plot artists for a specific frame in the animation.

		Inputs:
		- frame_id: Frame identifier to render.

		Outputs:
		- artists: List of matplotlib artists for blitting.
		"""
		for artist in player_dots + arrows + labels:
			artist.remove()
		player_dots.clear()
		arrows.clear()
		labels.clear()

		frame_df = location_data_single_play_df[location_data_single_play_df["frameId"] == frame_id]

		for _, row in frame_df.iterrows():
			x, y = row["x"], row["y"]
			dir_angle = row["dir"] if not np.isnan(row["dir"]) else 0
			dx = np.cos(np.radians(dir_angle))
			dy = np.sin(np.radians(dir_angle))

			if pd.isna(row["nflId"]):
				(dot,) = ax.plot(x, y, "o", color="brown", markersize=8)
				player_dots.append(dot)
			else:
				arrow = ax.arrow(x, y, dx, dy, head_width=1, color="blue", length_includes_head=True)
				label = ax.text(x + 0.5, y + 0.5, row["jerseyNumber"], fontsize=7, color="white")
				(dot,) = ax.plot(x, y, "o", color="blue", markersize=6)
				arrows.append(arrow)
				labels.append(label)
				player_dots.append(dot)

		# Show relative time from snap
		frame_offset = frame_id - snap_frame_id
		time_sec = frame_offset * 0.1  # 0.1 seconds per frame
		time_text.set_text(f"Time: {time_sec:+.1f} s (relative to snap)")

		# Down specification
		if down == 1:
			down_text = "1st"
		elif down == 2:
			down_text = "2nd"
		elif down == 3:
			down_text = "3rd"
		else:
			down_text = "4th"

		ax.set_title(
			f"Week {week}: {away_team} ({away_score}) vs {home_team} ({home_score}) \n Q{quarter} {down_text} and {yards_to_go} \n {play_description}",
		)

		return player_dots + arrows + labels + [time_text]

	ani = FuncAnimation(fig, update, frames=frame_ids, init_func=init, blit=False, interval=100)

	plt.show()
