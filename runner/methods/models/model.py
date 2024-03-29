import pandas as pd
from abc import ABC, abstractmethod, abstractproperty

from greenml.runner.methods.models.measure.pyrapl_measurement import PyRAPLMeasurement
# from measure.code_carbon_measurement import CodeCarbonMeasurement



class Model(ABC):
    """Machine learning model abstract class. It supplies the basic structure
    of our machine learning model.
    """

    def __init__(self, X_train, y_train, X_test, nb_folds, consumption_method):
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

        # get the measure method
        if consumption_method == None:
            self._measurement = None 
        elif consumption_method == "pyRAPL":
            self._measurement = PyRAPLMeasurement()
        elif consumption_method == "codecarbon":
            # self._measurement = CodeCarbonMeasurement()
            pass
        else:
            raise Exception("measure value does't refer an existing \
                measurement method")

    @abstractmethod
    def fit_cv(self):
        """Compute the predicted response vector given by the trained model.
        """
        pass

    @abstractmethod
    def predict(self):
        """Compute the predicted response vector given sklearn trained model.

        Returns:
            array: 1-D predicted response vector.
        """
        pass

    @abstractproperty
    def parameters(self) -> dict:
        pass

    @property
    def model_name(self) -> str:
        return type(self).__name__