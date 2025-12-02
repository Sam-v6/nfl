#!/usr/bin/env python

"""
Trains xgboost model on location tracking data
"""
# Standard imports
import logging
import os
import pickle
import time

import joblib

# Plotting imports
import matplotlib.pyplot as plt

# MLflow
import mlflow

# General imports
import numpy as np

# Local imports
from common.data_loader import DataLoader
from coverage.process_coverage import create_coverage_data
from sklearn.metrics import (
    classification_report,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# ML utils
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

# Models
from xgboost import XGBClassifier

from common.mlflow import setup_mlflow

# Local model imports
#from common.models.xgb import XGBModel
#from common.models.svc import SVC
from common.paths import PROJECT_ROOT


def load_data() -> dict[str]:
    base_path = PROJECT_ROOT / "data" / "coverage"
    data_file_list = ['x', 'y']
    data_dict = {}
    for file in data_file_list:
        file_name = file + '.pkl'
        file_path = base_path / file_name
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required data file {file_name} not found in {base_path}")
        else:
            with open(file_path, 'rb') as f:
                data_dict[file] = pickle.load(f)

    return data_dict

def plot_roc(y_test, y_proba) -> None:
    fpr, tpr, thresholds = roc_curve(y_test, y_proba[:, 1], pos_label=1)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_score(y_test, y_proba[:, 1]):.4f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    image_name = 'man_zone_roc_auc.png'
    image_path = PROJECT_ROOT / "output" / "coverage" / image_name
    plt.savefig(image_path, dpi=200)
    mlflow.log_artifact(image_path, artifact_path="plots")

def model_man_vs_zone() -> None:

    # Inputs
    RUN_DATA_PROCESSING = False

    ###########################################
    # Get data
    ###########################################
    # Run data processing if neccesary
    if RUN_DATA_PROCESSING:

        # Get raw data
        loader = DataLoader()
        games_df, plays_df, players_df, location_data_df = loader.get_data(weeks=[week for week in range (1,10)])

        # Process data
        create_coverage_data(games_df, plays_df, players_df, location_data_df)  # This will save down ML ready data to data/coverage/*.pkl

    # Get saved data locally
    data_dict = load_data()
    x_data = data_dict['x']
    y_data = data_dict['y']

    ###########################################
    # Split and standarize data
    ###########################################
    # Get x and y da
    logging.info("Scaling and saving training and test data...")

    # Splittys
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, stratify=y_data, random_state=42)

    # Standarize/scale data
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    ###########################################
    # Run training for models
    ###########################################
    # NOTES
    # - Consider oversampling/undersampling techniues like SMOTE, RandomOverSampler
    # - Organize the data into positions and then analyze feature importance
    # - Grid search on parameters
    # - Try additional tree based models XGBoost and LightGBM

    # Setup MLflow
    setup_mlflow(experiment_name="coverage-man-vs-zone")

    # Define 10-fold stratified cross-validation
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    # Xgb training
    xgbModel = XGBClassifier(
        objective="binary:logistic",
        n_estimators=1000,       # use more trees + early stopping when I add it later
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",      # fast histogram algo
        device="cuda",           # GPU
        random_state=42
    )
    
    # MLFlow wrapped training
    with mlflow.start_run(run_name="xgb-10fold"):
        # Log some dataset meta (sizes, class balance)
        mlflow.log_params({
            "cv_folds": 10,
            "test_size": 0.2,
            "shuffle": True,
            "random_state": 42
        })

        # Log class distribution
        _, counts = np.unique(y_train, return_counts=True)
        mlflow.log_metrics({
            "train_class0": int(counts[0]),
            "train_class1": int(counts[1]),
            "train_size":   int(x_train.shape[0]),
            "test_size":    int(x_test.shape[0]),
        })

        # Save scaler
        artifact_path = PROJECT_ROOT / "output" / "coverage"
        artifact_path.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, artifact_path / "standard_scaler.pkl")
        mlflow.log_artifact(artifact_path / "standard_scaler.pkl", artifact_path="preprocessing")

        # Enable autologging so the fitted model is captured
        mlflow.xgboost.autolog(log_model_signatures=True, log_input_examples=True)

        # Train
        train_xgb(
            model=xgbModel,
            skf=skf,
            x_train=x_train_scaled, y_train=y_train,
            x_test=x_test_scaled,   y_test=y_test
        )

def train_xgb(model, skf, x_train, y_train, x_test, y_test) -> None:
    start_time = time.time()

    logging.info("Starting K-Fold Cross-Validation...:")
    cv_scores = cross_val_score(model, x_train, y_train, cv=skf, scoring='roc_auc')

    # Log per-fold and summary CV AUC
    for i, s in enumerate(cv_scores, 1):
        mlflow.log_metric("cv_auc", float(s), step=i)
    mlflow.log_metric("cv_auc_mean", float(cv_scores.mean()))
    mlflow.log_metric("cv_auc_std",  float(cv_scores.std()))

    print(f"10-Fold Cross-Validation ROC AUC Scores: {cv_scores}")
    print(f"Mean ROC AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Fit on full training split (autolog will capture this model)
    model.fit(x_train, y_train)

    # Predict on test data
    y_pred   = model.predict(x_test)
    y_proba  = model.predict_proba(x_test)

    # Metrics on Test Data
    prec = precision_score(y_test, y_pred, pos_label=1)
    rec  = recall_score(y_test, y_pred, pos_label=1)
    f1   = f1_score(y_test, y_pred, pos_label=1)
    ll   = log_loss(y_test, y_proba)
    auc_ = roc_auc_score(y_test, y_proba[:, 1])

    print(f"Precision (Man): {prec:.4f}")
    print(f"Recall (Man): {rec:.4f}")
    print(f"F1 Score (Man): {f1:.4f}")
    print(f"Log Loss: {ll:.4f}")
    print(f"Overall ROC AUC: {auc_:.4f}")
    print(classification_report(y_test, y_pred, target_names=['Zone', 'Man']))

    # Log test metrics
    mlflow.log_metrics({
        "test_precision_pos1": float(prec),
        "test_recall_pos1":    float(rec),
        "test_f1_pos1":        float(f1),
        "test_log_loss":       float(ll),
        "test_auc":            float(auc_),
    })

    # Log the chosen hyperparameters as params (autolog also captures).
    try:
        mlflow.log_params(model.get_params())
    except Exception:
        pass

    # Save classification report as an artifact
    report_txt = classification_report(y_test, y_pred, target_names=['Zone', 'Man'])
    artifact_path = PROJECT_ROOT / "output" / "coverage"
    with open(artifact_path / "classification_report.txt", "w") as f:
        f.write(report_txt)
    mlflow.log_artifact(artifact_path / "classification_report.txt", artifact_path="reports")

    # Plot & log ROC curve
    plot_roc(y_test, y_proba)

    # Log model signature
    try:
        # Just grab a few vals which is sufficient for artifacts
        X_example = x_train[:5]
        y_example = y_train[:5]
        sig = mlflow.models.infer_signature(X_example, model.predict_proba(X_example))
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model_relogged",
            signature=sig,
            input_example=X_example
        )
    except Exception as e:
        logging.warning(f"Signature logging skipped: {e}")

    # Record training duration time
    end_time = time.time()
    training_duration = end_time - start_time
    logging.info(f"Model generation time: {training_duration:.2f} seconds")
    mlflow.log_metric("training_duration_sec", float(training_duration))

if __name__ == "__main__":
    
    # Configure basic logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Run model
    model_man_vs_zone()
