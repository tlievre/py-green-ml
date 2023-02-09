"""adaboost classifier script."""
from greenml.runner.methods.models.model import Model
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV


class Adaboost_Classifier(Model):
    """
    AdaBoost classification class inheriting from Model class.

    It uses AdaBoostClassifier to perform classification
    """

    def __init__(self, X_train, y_train, X_test, nb_folds, consumption_method,
                 params={
                     "n_estimators": [50, 100, 150, 200, 250],
                     "learning_rate": [0.0001, 0.001, 0.01, 0.1]
                     }
                 ):
        """Return object for class Adaboost_Classifier.

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
        self.__grid = GridSearchCV(AdaBoostClassifier(),
                                   params,
                                   cv=self._nb_folds,
                                   verbose=True)

    def fit_cv(self):
        """Compute the predicted response vector given\
        sklearn trained AdaBoost.

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
        given sklearn trained Adaboost.

        :returns: 1-D predicted response vector
        :rtype: np.array

        """
        return self.__grid.predict(self._X_test)

    @property
    def parameters(self) -> dict:
        return self.__parameters
