import pandas as pd
from abc import ABC, abstractmethod


class Model(ABC):

    def __init__(self, X_train, y_train, X_test):

        if not isinstance(X_train, pd.DataFrame):
            raise ValueError("X_train must be pandas.Dataframe")
        if not isinstance(y_train, pd.Series):
            raise ValueError("y_train must be pandas.Series")
        if not isinstance(X_test, pd.DataFrame):
            raise ValueError("X_test must be pandas.Dataframe")

        self.__X_train = X_train
        self.__y_train = y_train
        self.__X_test = X_test

    @abstractmethod
    def __fit_cv(self, nb_fold = 10):
        pass

    @abstractmethod
    def predict(self, nb_fold = 10):
        pass
