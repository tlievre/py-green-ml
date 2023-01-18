import pandas as pd
from abc import ABC, abstractmethod


class Model(ABC):

    def __init__(self, X_train, y_train, X_test, nb_folds):

        if not isinstance(X_train, pd.DataFrame):
            raise ValueError("X_train must be pandas.Dataframe")
        if not isinstance(y_train, pd.Series):
            raise ValueError("y_train must be pandas.Series")
        if not isinstance(X_test, pd.DataFrame):
            raise ValueError("X_test must be pandas.Dataframe")

        self.__X_train = X_train
        self.__y_train = y_train
        self.__X_test = X_test
        self.__nb_folds = nb_folds

    @abstractmethod
    def __fit_cv(self):
        pass

    @abstractmethod
    def predict(self):
        pass
