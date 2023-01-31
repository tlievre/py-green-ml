from greenml.runner.methods.models.model import Model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

import numpy as np

class Logistic_Regression(Model):
    """Logistic regression linear model implemetend from sklearn.
    Inherit from Model abstract class
    """

    def __init__(self, X_train, y_train, X_test, nb_folds,
        params = {
            'tol': np.linspace(1e-5, 1e-2, 5),
            'C': np.linspace(1e-3, 1, 5),
            'penalty' : ['None', 'l2', 'l1', 'elasticnet'],
            'solver' : ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']}
            # The choice of the algorithm depends on the penalty chosen. Supported penalties by solve
        ):
        """
        Args:
            X_train (pd.DataFrame): Train set predictors.
            y_train (pd.DataFrame): Train set responses.
            X_test (pd.DataFrame): Test set predictors.
            nb_folds (int): Folds numbers used in cross validation.
            params (dict, optional): Contains listes of tuning parameters given
            by sklearn Logistic_Regression() model. Defaults to 
                { 
                    'tol': np.linspace(1e-5, 1e-2, 5),
                    'C': np.linspace(1e-3, 1, 5),
                    'penalty' : ['None', 'l2', 'l1', 'elasticnet'], 
                    'solver' : ['lbfgs', 'liblinear', 'newton-cg','newton-cholesky', 'sag', 'saga']
                }.
        """
        super().__init__(X_train, y_train, X_test, nb_folds)
        self.__parameters = params
        self.__grid = GridSearchCV(LogisticRegression(), params,
            cv = self._nb_folds, verbose = True)

    def fit_cv(self):
        """Compute the predicted response vector given by the trained
        logistic regression.

        Returns:
            float: measure or None.
        """
        if self.__measurement is None:
            return_value = None
        else :
            # training measure
            self.__measurement.begin()
            self.__grid.fit(self._X_train, self._y_train)
            self.__measurement.end()
            return_value = self.__measurement.convert()
        return return_value

    def predict(self):
        """Compute the predicted response vector given sklearn trained model
        logistic regression.

        Returns:
            array: 1-D predicted response vector.
        """
        return self.__grid.predict(self._X_test)

    @property
    def parameters(self) -> dict:
        return self.__parameters