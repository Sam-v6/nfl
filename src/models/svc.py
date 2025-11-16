#!/usr/bin/env python

"""
Contains class for default SVC model
"""

from sklearn.svm import SVC

models['svm'] = SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42)