#!/usr/bin/env python

"""
Module: data.py
Description: Class for loading raw tracking data

Author: Syam Evani
Created: 2025-10-15
"""

import xgboost as xgb

class XGBModel():
    def __init__(self, objective='binary:logistic', n_estimators=100, random_state=42):
        self.model = xgb.XGBClassifier(objective=objective, n_estimators=n_estimators, random_state=random_state)