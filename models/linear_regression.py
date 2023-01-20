from greenml.models.model import Model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV


class Linear_Regression(Model):

    def __init__(self, X_train, y_train, X_test, nb_folds,
        params = {
            'positive' : [True, False],
            'fit_intercept' : [True, False]} # not necessary
    ):
        super().__init__(X_train, y_train, X_test, nb_folds)
        self._hparam = params

    def _fit_cv(self):
        grid = GridSearchCV(LinearRegression(), self._hparam, cv = self._nb_folds, verbose = True)
        grid.fit(self._X_train, self._y_train)
        return grid