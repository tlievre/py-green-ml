from greenml.runner.methods.models.model import Model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV


class Knn_Classification(Model):
    """
    Knn classification class inheriting from Model class.
    It uses KNeighborsClassifier to perform classification
    """

    def __init__(self, X_train, y_train, X_test, nb_folds, consumption_method,
                 params={
                     "n_neighboors"=[1, 3, 5, 7, 11]
                     }
                 ):
        super().__init__(X_train, y_train, X_test,
                         nb_folds, consumption_method)
        self.__parameters = params
        self.__grid = GridSearchCV(KNeighborsClassifier(),
                                   params,
                                   cv=self._nb_folds,
                                   verbose=True)

    def fit_cv(self):
        if self.measurement is None:
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
        return self.__grid.predict(self._X_test)

    @property
    def parameters(self) -> dict:
        return self.__parameters


class Knn_Regression(Model):
    def __init__(self, X_train, y_train, X_test, nb_folds,
                 consumption_method,
                 params = {
                     "n_neighbors" = [1, 3, 5, 7, 11]
                     }
                 ):
        super().__init__(X_train, y_train, X_test, nb_folds,
                         consumption_method)
        self.__parameters = params
        self.__grid = GridSearchCV(KNeighborsRegressor(),
                                   params,
                                   cv=self._nb_folds,
                                   verbose = True)

    def fit_cv(self):
        if self.measure is None:
            self.__grid.fit(self._X_train, self._y_train)
            return_value = None
        else:
            # measurement training
            self._measurement_begin()
            self.__grid.fit(self._X_train, self._y_train)
            self._measurement.end()
            return_value = self._measurement.convert()
        return return_value

    def predict(self):
        return self.__grid.predict(self._X_test)

    @property
    def parameters(self) -> dict:
        return self.__parameters
        
