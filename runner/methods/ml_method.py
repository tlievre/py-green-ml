# coding: utf-8
import importlib
from abc import ABC, abstractmethod


class ML_method(ABC) :
    """This abstract class implement the basic structure of our machine
    learning statistic methods (classification, regression, clustering ...).
    """
    
    def __init__(self, data_train, data_test, token, nb_folds) :
        """
        Args:
            data_train (_type_): _description_
            data_test (_type_): _description_
            token (_type_): _description_
            nb_folds (_type_): _description_

        Raises:
            ValueError: _description_
        """

        # train/test split
        self._X_train = data_train.loc[:, data_train.columns != self._data_y_var] # ?? for clust
        self._y_train = data_train[self._data_y_var]
        self._X_test = data_test.loc[:, data_test.columns != self._data_y_var]
        self._y_test = data_test[self._data_y_var]

        self._nb_folds = nb_folds

        # get the suitable method class
        try :
            module_path = "greenml.models." + token
            models = importlib.import_module(module_path)
            constr_model = getattr(models, token)
            self._model = constr_model(self._X_train, self._y_train, self._X_test, self._nb_folds)
        except (ImportError, AttributeError):
            raise ValueError("Unknown " + module_path)

    
    def fit(self) :
        """This protected method aim to construct the suitable model and method. 

        Args:
            nb_folds (int, optional): Set the folds numbers in the cross validation. Defaults to 10.
        """
        
        return self._model.fit_cv()

    @abstractmethod
    def get_metrics(self):
        """compute the metrics following the suitable inherited method class
        """
        pass