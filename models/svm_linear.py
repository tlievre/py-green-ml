import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV

from greenml.models.model import Model

class SVM_Linear(Model):

    # params need to be test
    def __init__(self, X_train, y_train, X_test, nb_folds ,
        params = {
            'tol': np.linspace(1e-5, 1e-2, 5),
            'C': np.linspace(1e-3, 1, 5)
        }):

        super().__init__(X_train, y_train, X_test, nb_folds)

        self._hparam = params

    def _fit_cv(self):
        grid = GridSearchCV(LinearSVC(), self._hparam, cv = self._nb_folds, verbose = True)
        grid.fit(self._X_train, self._y_train)
        return grid