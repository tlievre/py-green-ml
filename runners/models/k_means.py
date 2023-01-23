from greenml.models.model import Model
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV

class K_Means(Model):

    def __init__(self, X_train, y_train, X_test, nb_folds,
        params = {
            'n_clusters' : [i for i in range(1, 20)], # /!\ max clusters
            'algorithm' : ["loyd", "elkan"] # auto and full deprecated

        }):
        """
        Args:
           X_train (pd.DataFrame): Train set predictors.
            y_train (pd.DataFrame): Train set responses.
            X_test (pd.DataFrame): Test set predictors.
            nb_folds (int): Folds numbers used in cross validation.
            params (dict, optional): Contains listes of tuning parameters given
            by sklearn KMeans() model. Defaults to
                {
                    'n_clusters' : [i for i in range(1, 20)],
                    'algorithm' : ["loyd", "elkan"]
                }.
        """
        super().__init__(X_train, y_train, X_test, nb_folds)
        self._hparam = params
    
    def _fit_cv(self):
        """Compute the predicted response vector given sklearn trained KMeans.

        Returns:
            array: 1-D predicted repsonse vector
        """
        grid = GridSearchCV(KMeans(n_init = 'auto'), self._hparam, cv = self._nb_folds, verbose = True)
        grid.fit(self._X_train, self._y_train)
        return grid