from sklearn.ensemble import RandomForestClassifier

class RFRModel():
    def __init__(self, n_estimators=100, class_weight='balanced', random_state=42):
        self.model = RandomForestClassifier(n_estimators=n_estimators, class_weight=class_weight, random_state=random_state)