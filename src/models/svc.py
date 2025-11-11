#!/usr/bin/env python

"""
Module: data.py
Description: Class for loading raw tracking data

Author: Syam Evani
Created: 2025-10-15
"""
from sklearn.svm import SVC

models['svm'] = SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42)