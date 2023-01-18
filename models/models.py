from logging import warning
from sklearn.model_selection import GridSearchCV
import pandas as pd
from abc import ABC

# Will be raised when model affect in parameter don't inherit from
# our models template
class ErrorMethod(Exception):
    pass

class Model(ABC):

    def __init__(self, X_train, y_train, X_test):

        if not isinstance(X_train, pd.DataFrame):
            raise ValueError("X_train must be pandas.Dataframe")
        if not isinstance(y_train, pd.Series):
            raise ValueError("y_train must be pandas.Series")
        if not isinstance(X_test, pd.DataFrame):
            raise ValueError("X_test must be pandas.Dataframe")

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        
        self.hparam = {}

        self.model = 0

        self.y_pred = None

    def fit_CV(self, nb_fold=10):
        
        if isinstance(self.model, int):
            raise ErrorMethod("fit_CV must be used on object from inherited class from module greenml.models")
        else:

            grid = GridSearchCV(self.model, self.hparam, cv=nb_fold, verbose=True)
            grid.fit(self.X_train, self.y_train)
            self.y_pred = grid.best_estimator_.predict(self.X_test)

            return self.y_pred
