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

