from greenml.models.model import Model
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV

# /!\ doesn't work, need to be fixed
class Naive_Bayes(Model):

    def __init__(self, X_train, y_train, X_test, nb_folds,
        params = {'alpha': [0.00001, 0.0001, 0.001, 0.1, 1, 10, 100, 1000]}):

        super().__init__(X_train, y_train, X_test, nb_folds)
        self._hparam = params

    def _fit_cv(self):
        grid = GridSearchCV(MultinomialNB(), self._hparam, cv = self._nb_folds, verbose = True)
        grid.fit(self._X_train, self._y_train)
        return grid