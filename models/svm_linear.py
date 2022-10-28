import numpy as np
from greenml.runners.models import Model

class SVM_Linear(Model):

    def __init__(self, X_train, y_train, X_test):

        super().__init__(X_train, y_train, X_test)

        from sklearn.svm import LinearSVC

        self.method = LinearSVC()

        self.hparam = {
            "tol": np.linspace(1e-5, 1e-2, 5),
            "C": np.linspace(1e-3, 1, 5)
        }