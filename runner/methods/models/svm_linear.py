import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV

from greenml.runner.methods.models.model import Model

class SVM_Linear(Model):
    """Linear support vector machine class. It uses LinearSVC() from
    sklearn.svm. Inherit from Model abstract class.
    """

    # params need to be test
    def __init__(self, X_train, y_train, X_test, nb_folds, consumption_method,
        params = {
            'dual':[False,True],
            'C': [0.1, 1, 10, 100, 1000],
            'max_iter' : [1000,2000,5000]
        }):
        """
        Args:
            X_train (pd.DataFrame): Train set predictors.
            y_train (pd.DataFrame): Train set responses.
            X_test (pd.DataFrame): Test set predictors.
            nb_folds (int): Folds numbers used in cross validation.
            params (dict, optional): Contains listes of tuning parameters given
            by sklearn LinearSVC() model. Defaults to
                {
                    'dual':[False,True],
                    'C': [0.1, 1, 10, 100, 1000],
                    'max_iter' : [1000,2000,5000]
                }.
        """
        super().__init__(X_train, y_train, X_test, nb_folds,
            consumption_method)

        self.__parameters = params
        self.__grid = GridSearchCV(LinearSVC(), params,
            cv = self._nb_folds, verbose = True)

    def fit_cv(self):
        """Compute the predicted response vector given by the trained model
        LinearSVC().

        Returns:
            float: measure.
        """
        if self._measurement is None:
            self.__grid.fit(self._X_train, self._y_train)
            return_value = None
        else :
            # training measure
            self._measurement.begin()
            self.__grid.fit(self._X_train, self._y_train)
            self._measurement.end()
            return_value = self._measurement.convert()
        return return_value
    
    def predict(self):
        """Compute the predicted response vector given sklearn trained model
        LinearSVC().

        Returns:
            array: 1-D predicted response vector.
        """
        return self.__grid.predict(self._X_test)

    @property
    def parameters(self) -> dict:
        return self.__parameters