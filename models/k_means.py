from greenml.models.model import Model
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV

class K_Means(Model):

    def __init__(self, X_train, y_train, X_test, nb_folds,
        params = {
            'n_clusters' : [i for i in range(1, 20)], # /!\ max clusters
            'algorithm' : ["loyd", "elkan"] # auto and full deprecated

        }):
        super().__init__(X_train, y_train, X_test, nb_folds)
        self._hparam = params
    
    def _fit_cv(self):
        grid = GridSearchCV(KMeans(n_init = 'auto'), self._hparam, cv = self._nb_folds, verbose = True)
        grid.fit(self._X_train, self._y_train)
        return grid