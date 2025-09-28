#/usr/bin/env python

"""
Purpose: Process nfl data for machine learning model creation
Author: Syam Evani
Date: April 2025
"""

# Standard imports
import os
import random
import time
import pickle
import logging

# General imports
import numpy as np
import pandas as pd

# Plotting imports
import seaborn as sns
import matplotlib.pyplot as plt

# ML utils
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict, train_test_split
from sklearn.metrics import roc_auc_score, log_loss, precision_score, recall_score, f1_score, classification_report, roc_curve, RocCurveDisplay
from sklearn.preprocessing import StandardScaler

# Model imports
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

# Local imports
from common.data_loader import DataLoader
from coverage.process_coverage import create_coverage_data

def create_models(x_train, y_train, x_test, y_test):

    #--------------------------------------------------
    # Build models
    #--------------------------------------------------
    # NOTES
    # - Consider oversampling/undersampling techniues like SMOTE, RandomOverSampler
    # - Organize the data into positions and then analyze feature importance
    # - Grid search on parameters
    # - Try additional tree based models XGBoost and LightGBM
    # - Save data out that I can load it later

    model_data_start = time.time()
    print("----------------------------------------------------------")
    print("STATUS: Bulding models...")

    models = {}
    models['log'] = LogisticRegression(max_iter=1000, class_weight='balanced')
    models['rfr'] = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    models['xgb'] = xgb.XGBClassifier(objective='binary:logistic', n_estimators=100, random_state=42)
    #models['knn'] = KNeighborsClassifier(n_neighbors=5)
    #models['svm'] = SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42)
    models['lgb'] = lgb.LGBMClassifier(n_estimators=100, class_weight='balanced', random_state=42)

    # Define 10-fold stratified cross-validation
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    for model_name, model in models.items():
        print("----------------------------------------------------------")
        print(f"Model: {model_name}")

        if model_name == 'lgb':
            x_test = pd.DataFrame(x_test)
            x_train = pd.DataFrame(x_train)
            y_train = pd.Series(y_train).ravel()
            y_test = pd.Series(y_test).ravel()

        # K Fold validation
        print("K-Fold Cross-Validation on Training Data:")
        cv_scores = cross_val_score(model, x_train, y_train, cv=skf, scoring='roc_auc')
        print(f"10-Fold Cross-Validation ROC AUC Scores: {cv_scores}")
        print(f"Mean ROC AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        # Fit on training data
        model.fit(x_train, y_train)

        # Predict on test data
        y_pred = model.predict(x_test)
        y_proba = model.predict_proba(x_test)

        # Metrics on Test Data
        print(f"Precision (Man): {precision_score(y_test, y_pred, pos_label=1):.4f}")
        print(f"Recall (Man): {recall_score(y_test, y_pred, pos_label=1):.4f}")
        print(f"F1 Score (Man): {f1_score(y_test, y_pred, pos_label=1):.4f}")
        print(f"Log Loss: {log_loss(y_test, y_proba):.4f}")
        print(f"Overall ROC AUC: {roc_auc_score(y_test, y_proba[:, 1]):.4f}")
        print(classification_report(y_test, y_pred, target_names=['Zone', 'Man']))

        # Plot ROC AUC Curve
        fpr, tpr, thresholds = roc_curve(y_test, y_proba[:, 1], pos_label=1)
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_score(y_test, y_proba[:, 1]):.4f})')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name.upper()}')
        plt.legend(loc='lower right')
        plt.grid(True)
        image_name = f'man_zone_{model_name}_roc_auc.png'
        plt.savefig(os.path.join(os.getenv('NFL_HOME'), 'output', 'coverage', image_name))

    model_data_end = time.time()
    print(f"Model generation time: {model_data_end - model_data_start:.2f} seconds")

    # Return 
    return 0

def load_data():
    base_path = os.path.join(os.getenv('NFL_HOME'), 'data', 'coverage')
    data_file_list = ['x', 'y']
    data_dict = {}
    for file in data_file_list:
        file_name = file + '.pkl'
        file_path = os.path.join(base_path, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required data file {file_name} not found in {base_path}")
        else:
            with open(file_path, 'rb') as f:
                data_dict[file] = pickle.load(f)

    return data_dict
    
def model_man_vs_zone():

    # Inputs
    RUN_DATA_PROCESSING = True

    # Run data processing if neccesary
    if RUN_DATA_PROCESSING:

        # Get raw data
        loader = DataLoader()
        games_df, plays_df, players_df, location_data_df = loader.get_data(weeks=[week for week in range (1,10)])

        # Process data
        create_coverage_data(games_df, plays_df, players_df, location_data_df)  # This will save down ML ready data to data/coverage/*.pkl

    # Get saved data locally
    data_dict = load_data()

    logging.info("Scaling and saving training and test data...")

    # Splittys
    x_train, x_test, y_train, y_test = train_test_split(data_dict['x'], data_dict['y'], test_size=0.2, stratify=y_array, random_state=42)

    # Standarize/scale data
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Create model

    # Create models
    create_models(x_train_scaled, y_train, x_test_scaled, y_test)

    # Return
    return 0

if __name__ == "__main__":
    
    # Configure basic logging (optional, but useful for quick setup)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Get a logger instance
    logger = logging.getLogger(__name__)

    # Run model
    model_man_vs_zone()
