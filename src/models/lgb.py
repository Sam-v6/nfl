#!/usr/bin/env python

"""
Module: data.py
Description: Class for loading raw tracking data

Author: Syam Evani
Created: 2025-10-15
"""

import lightgbm as lgb

class LGBModel():
    def __init__(self, n_estimators=100, class_weight='balanced', random_state=42):
        self.model = lgb.LGBMClassifier(n_estimators=n_estimators, class_weight=class_weight, random_state=random_state)