#/usr/bin/env python

"""
Purpose: Process nfl data for machine learning model creation
Author: Syam Evani
Date: April 2025
"""

# Standard imports
import os

# Additional imports
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd

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

def animate_play(week_df, game_id, play_id):
    game_id = week_df['gameId'].iloc[0]
    play_id = week_df[week_df['gameId'] == game_id]['playId'].iloc[0]
    play_df = week_df[(week_df['gameId'] == game_id) & (week_df['playId'] == play_id)]

    frame_ids = sorted(play_df['frameId'].unique())

    # 🔍 Find the frameId of the snap
    snap_row = play_df[play_df['event'] == 'ball_snap']
    snap_frame_id = snap_row['frameId'].min() if not snap_row.empty else frame_ids[0]

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 53.3)
    ax.set_facecolor('green')
    ax.set_aspect('equal')

    for x in range(10, 111, 10):
        ax.axvline(x, color='white', linewidth=1)
    ax.axhline(0, color='white')
    ax.axhline(53.3, color='white')

    player_dots = []
    arrows = []
    labels = []

    # ⏱️ Placeholder for relative time display
    time_text = ax.text(60, 51, '', fontsize=12, ha='center', color='white', bbox=dict(facecolor='black', alpha=0.5))

    def init():
        return player_dots + arrows + labels + [time_text]

    def update(frame_id):
        for artist in player_dots + arrows + labels:
            artist.remove()
        player_dots.clear()
        arrows.clear()
        labels.clear()

        frame_df = play_df[play_df['frameId'] == frame_id]

        for _, row in frame_df.iterrows():
            x, y = row['x'], row['y']
            dir_angle = row['dir'] if not np.isnan(row['dir']) else 0
            dx = np.cos(np.radians(dir_angle))
            dy = np.sin(np.radians(dir_angle))

            if pd.isna(row['nflId']):
                dot, = ax.plot(x, y, 'o', color='brown', markersize=8)
                player_dots.append(dot)
            else:
                arrow = ax.arrow(x, y, dx, dy, head_width=1, color='blue', length_includes_head=True)
                label = ax.text(x + 0.5, y + 0.5, row['jerseyNumber'], fontsize=7, color='white')
                dot, = ax.plot(x, y, 'o', color='blue', markersize=6)
                arrows.append(arrow)
                labels.append(label)
                player_dots.append(dot)

        # ⏱️ Show relative time from snap
        frame_offset = frame_id - snap_frame_id
        time_sec = frame_offset * 0.1  # 0.1 seconds per frame
        time_text.set_text(f"Time: {time_sec:+.1f} s (relative to snap)")
        
        ax.set_title(f'Game {game_id}, Play {play_id}, Frame {frame_id}')
        
        return player_dots + arrows + labels + [time_text]

    ani = FuncAnimation(fig, update, frames=frame_ids, init_func=init,
                        blit=False, interval=100)

    plt.show()