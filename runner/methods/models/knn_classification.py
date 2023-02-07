"""script for knn classification."""

from greenml.runner.methods.models.model import Model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


class Knn_Classification(Model):
    """
    Knn classification class inheriting from Model class.

    It uses KNeighborsClassifier to perform classification
    """

    def __init__(self, X_train, y_train, X_test, nb_folds, consumption_method,
                 params={
                     "n_neighbors": [1, 3, 5, 7, 11]
                     }
                 ):
        """Return object for class Knn_Regression.

        :param pd.DataFrame X_train: Train set predictors.
        :param pd.DataFrame y_train: Train set responses.
        :param X_test: Test set predictors.
        :param int nb_folds: Folds number used in cross validation.
        :param string consumption_method: Algorithm to use for.
        :param dict params: parameters for cross validation.
        :returns:
        :rtype:

        """
        super().__init__(X_train, y_train, X_test,
                         nb_folds, consumption_method)
        self.__parameters = params
        self.__grid = GridSearchCV(KNeighborsClassifier(),
                                   params,
                                   cv=self._nb_folds,
                                   verbose=True)

    def fit_cv(self):
        """Compute the predicted response vector given\
        sklearn trained knn.

        :returns: measure
        :rtype: float

        """
        if self._measurement is None:
            self.__grid.fit(self._X_train, self._y_train)
            return_value = None
        else:
            # Measure the consumption
            self._measurement.begin()
            self.__grid.fit(self._X_train, self._y_train)
            self._measurement.end()
            return_value = self._measurement.convert()
        return return_value

    def predict(self):
        """Compute the predicted response vector\
        given sklearn trained Knn.

        :returns: 1-D predicted response vector
        :rtype: np.array

        """
        return self.__grid.predict(self._X_test)

    @property
    def parameters(self) -> dict:
        return self.__parameters
