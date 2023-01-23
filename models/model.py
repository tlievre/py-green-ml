import pandas as pd
from abc import ABC, abstractmethod


class Model(ABC):
    """Machine learning model abstract class. It supplies the basic structure
    of our machine learning model.
    """

    def __init__(self, X_train, y_train, X_test, nb_folds):
        """
        Args:
            X_train (pd.DataFrame): Train set predictors.
            y_train (pd.DataFrame): Train set responses.
            X_test (pd.DataFrame): Test set predictors.
            nb_folds (int): Folds numbers used in cross validation.

        Raises:
            ValueError: X_train type error
            ValueError: y_train type error
            ValueError: X_test type error
        """
        if not isinstance(X_train, pd.DataFrame):
            raise ValueError("X_train must be pandas.Dataframe")
        if not isinstance(y_train, pd.Series):
            raise ValueError("y_train must be pandas.Series")
        if not isinstance(X_test, pd.DataFrame):
            raise ValueError("X_test must be pandas.Dataframe")

        self._X_train = X_train
        self._y_train = y_train
        self._X_test = X_test
        self._nb_folds = nb_folds

    @abstractmethod
    def _fit_cv(self):
        pass

    def predict(self):
        """Compute the predicted response vector given sklearn trained model.

        Returns:
            array: 1-D predicted response vector.
        """
        grid = self._fit_cv()
        return grid.best_estimator_.predict(self._X_test)