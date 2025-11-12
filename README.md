
# Overview
This repo trains on the 2022 NFL Season Weeks 1 - 9 to predict man or zone pass coverage with machine learning. 

Two models are implemented:
- LSTM "naive" model that treats location tracking data as true time series inputs
- Transformer model that follows the SumerSports and SmitBajaj implementation

Both models yield about 85% accuracy in prediciting man or pass coverage for all games in Week 9 of the 2022 NFL season.

# Demo

![Bengals Play Animation](./docs/videos/bengals_plot.gif)

# Getting started

## Get the raw data
The raw data that was provided by NFL NextGenStats has been removed from the original Kaggle source. The orginal csv data has been transformed to parquet data and stored with git large file storage (LFS).

To retrieve the data, first install git LFS

```bash
# Install git lfs
sudo apt install git-lfs

# Navigate to where you cloned this repo
cd some-path/nfl

# Seperately run a pull with lfs
git lfs install
git lfs pull
```

This will fully populate `nfl/parquet/*` with:
- `games.parquet`
- `player_play.parquet`
- `players.parquet`
- `plays.parquet`
- `tracking_week_1.parquet`
- `tracking_week_2.parquet`
- `tracking_week_3.parquet`
- `tracking_week_4.parquet`
- `tracking_week_5.parquet`
- `tracking_week_6.parquet`
- `tracking_week_7.parquet`
- `tracking_week_8.parquet`
- `tracking_week_9.parquet`


## Setup python virtual environment
This project uses `uv` for managing the python virtual environment. To install uv please see the official [documentation](https://docs.astral.sh/uv/getting-started/installation/) or install via CLI below:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

To create and activate the virtual environment, run the following:
```bash
uv sync --locked
source .venv/bin/activate
```

## Activating MLflow
This project uses MLflow to manage ML models and track experiments from hyper paremter optimization. To start MLflow simply:
```bash
# Make sure your virtual environment is started if you opened a new terminal
source .venv/bin/activate

# Start the tracking server
mlflow ui --backend-store-uri ./mlruns
```

The last command will start the MLflow GUI at your local host loopback on port 5000: http://127.0.0.1:5000


# Development Roadmap
1) Naive tree-based mdoels and simple MLP on last frame data before play (not time series)
2) LSTM/GRU with time series
3) Temporal CNNs
4) Temporal Transformer
5) Spatio-temporal GNN

# TODO
### Update Github CI to use LFS
- uses: actions/checkout@v4
  with:
    lfs: true

variables:
  GIT_LFS_SKIP_SMUDGE: "0"   # ensure LFS files are fetched
before_script:
  - git lfs install
  - git lfs fetch
  - git lfs checkout

#### Plots

# Do these generally but also per team 
coverage vs yards gained --> horizontal box plot
coverage vs down and distance --> heat map?, 

coverage ran vs team --> heat map