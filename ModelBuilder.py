from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class ModelBuilder:
    def __init__(self, model, test_size=0.2, random_state=42):
        self.model = model
        self.test_size = test_size
        self.random_state = random_state

    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
        self.model.fit(X_train, y_train)
        self.X_test = X_test
        self.y_test = y_test

    def evaluate(self):
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        return accuracy

    def predict(self, X):
        return self.model.predict(X)