"""Python script for xgb regressor."""
from greenml.runner.methods.models.model import Model
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV


class Xgb_Regressor(Model):
    """
    Xgb_regressor model class inheriting from Model Class.

    It uses KNeighborsClassifier to perform regression.
    """

    def __init__(self, X_train, y_train, X_test,
                 nb_folds, consumption_method,
                 params={
                     'learning_rate': [0.0001, 0.001, 0.01, 0.3],
                     'n_estimators': [100, 150, 200, 250],
                     'max_depth': [3, 5, 6]
                     }):
        """
        Return object for xgboost regressor.

        :param pd.DataFrame X_train: Train set predictors.
        :param pd.DataFrame y_train: Train set responses.
        :param pd.DataFrme X_test: Test set predictors
        :param int nb_folds: Folds number used in cross validation.
        :param string consumption_method: Algorithm to use for measurement.
        :param dict params: Parameters for cross validation.
        :returns:
        :rtype:

        """
        super().__init__(X_train, y_train, X_test,
                         nb_folds, consumption_method)
        self.__parameters = params
        self.__grid = GridSearchCV(XGBRegressor(), params,
                                   cv=self._nb_folds, verbose=True)

        def fit_cv(self):
            """Compute the predicted response vector\
            given by the trained xgboost regressor.

            :returns:
            :rtype:

            """
            if self._measurement is None:
                self.__grid.fit(self._X_train, self._y_train)
                return_value = None
            else:
                # training measure
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
