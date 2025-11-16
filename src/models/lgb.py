#!/usr/bin/env python

"""
Contains class for light gradient boosting model for general usage
"""

import lightgbm as lgb

class LGBModel():
    def __init__(self, n_estimators=100, class_weight='balanced', random_state=42):
        self.model = lgb.LGBMClassifier(n_estimators=n_estimators, class_weight=class_weight, random_state=random_state)