from sklearn.neighbors import KNeighborsClassifier

class KNNModel():
    def __init__(self, n_neighbors=5):
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)

