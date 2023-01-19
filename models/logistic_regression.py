from greenml.models.model import Model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

import numpy as np

class LogisticRegression(Model):

    def __init__(self, X_train, y_train, X_test, nb_folds,
        params = {
            'tol': np.linspace(1e-5, 1e-2, 5),
            'C': np.linspace(1e-3, 1, 5),
            'penalty' : ['None', 'l2', 'l1', 'elasticnet'],
            'solver' : ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']}
            # The choice of the algorithm depends on the penalty chosen. Supported penalties by solve
        ):
        super().__init__(X_train, y_train, X_test, nb_folds)
        self._hparam = params

    def _fit_cv(self):
        grid = GridSearchCV(LogisticRegression(), self._hparam, cv = self._nb_folds, verbose = True)
        grid.fit(self._X_train, self._y_train)
        return grid