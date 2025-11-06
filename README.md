# nfl
Machine learning sandbox for nfl data

`uv compile pyproject.toml -o requirements.txt`


`mlflow ui --backend-store-uri ./mlruns`

## General developement approach
1) Naive tree-based mdoels and simple MLP on last frame data before play (not time series)
2) LSTM/GRU with time series
3) Temporal CNNs
4) Temporal Transformer
5) Spatio-temporal GNN

# TODO
## Update Gitlab CI to use LFS
- uses: actions/checkout@v4
  with:
    lfs: true


variables:
  GIT_LFS_SKIP_SMUDGE: "0"   # ensure LFS files are fetched
before_script:
  - git lfs install
  - git lfs fetch
  - git lfs checkout


# Getting Started


## Get the raw data
The raw tracking data that was provided by NFL NextGenStats has been removed from the original Kaggle source. The orginal csv data has been transformed to parquet data and stored with git large file storage (LFS).

To retrieve the data, first install git LFS

```bash
# Install git lfs
sudo apt install git-lfs

# Navigate to where you cloned this repo
# Seperately run a pull with lfs
git lfs install
git lfs pull


# Plots

# Do these generally but also per team 
coverage vs yards gained --> horizontal box plot
coverage vs down and distance --> heat map?, 

coverage ran vs team --> heat map