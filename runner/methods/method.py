# coding: utf-8
import importlib
from abc import ABC, abstractmethod


class Method(ABC) :
    """This abstract class implement the basic structure of our machine
    learning statistic methods (classification, regression, clustering ...).
    """
    
    def __init__(self, data_train, data_test, data_y_var, models_name, nb_folds) :
        """

        Attributes:
            data_train (obj): Train data, store in a pandas.DataFrame object.
            data_test (obj): test data, store in a pandas.DataFrame object.
            data_y_var (str): Column name of the response variable.
            models_name (list): List of models (similar classes) names.
            nb_folds (int): Folds numbers. 
        """

        # train/test split
        self._X_train = data_train.loc[:, data_train.columns != data_y_var] # ?? for clust
        self._y_train = data_train[data_y_var]
        self._X_test = data_test.loc[:, data_test.columns != data_y_var]
        self._y_test = data_test[data_y_var]
        self._nb_folds = nb_folds
        
        # import
        for model_name in models_name:
            module_name = model_name.lower()
            # get the suitable method class
            try :
                module_path = "greenml.runner.methods.models." + module_name
                module = importlib.import_module(module_path)
                constr_model = getattr(module, model_name)
                self._model = constr_model(self._X_train, self._y_train, self._X_test, self._nb_folds)
            except (ImportError, AttributeError):
                raise ValueError("Unknown " + module_path)

    
    def fit(self) :
        """This protected method aim to construct the suitable model and method. 

        Args:
            nb_folds (int, optional): Set the folds numbers in the cross validation. Defaults to 10.
        """
        self._model.fit_cv()

    @abstractmethod
    def get_metrics(self):
        """compute the metrics following the suitable inherited method class
        """
        pass