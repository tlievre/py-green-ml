from greenml.runner.methods.models.model import Model
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
        self.__parameters = params
        self.__grid = GridSearchCV(KMeans(n_init = 'auto'), params,
            cv = self._nb_folds, verbose = True)
    

    def fit_cv(self):
        """Compute the predicted response vector given sklearn trained KMeans.

        Returns:
            array: 1-D predicted repsonse vector
        """
        self.__grid.fit(self._X_train, self._y_train)


    def predict(self):
        """Compute the predicted response vector given sklearn trained model.

        Returns:
            array: 1-D predicted response vector.
        """
        return self.__grid.predict(self._X_test)
    

    @property
    def parameters(self) -> dict:
        return self.__parameters