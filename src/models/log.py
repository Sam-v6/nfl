#!/usr/bin/env python

"""
Module: data.py
Description: Class for loading raw tracking data

Author: Syam Evani
Created: 2025-10-15
"""

from sklearn.linear_model import LogisticRegression

class LogisticModel():
    def __init__(self, max_iter=1000, class_weight='balanced'):
        self.model = LogisticRegression(max_iter=max_iter, class_weight=class_weight)