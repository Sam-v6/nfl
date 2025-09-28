import xgboost as xgb

class XGBModel():
    def __init__(self, objective='binary:logistic', n_estimators=100, random_state=42):
        self.model = xgb.XGBClassifier(objective=objective, n_estimators=n_estimators, random_state=random_state)