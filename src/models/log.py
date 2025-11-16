#!/usr/bin/env python

"""
Contains class for default sklearn logisitc regression model for general usage
"""

from sklearn.linear_model import LogisticRegression

class LogisticModel():
    def __init__(self, max_iter=1000, class_weight='balanced'):
        self.model = LogisticRegression(max_iter=max_iter, class_weight=class_weight)