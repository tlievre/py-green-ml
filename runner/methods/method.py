# coding: utf-8
import importlib
from abc import ABC, abstractmethod

from sklearn.exceptions import NotFittedError


class Method(ABC) :
    """This abstract class implement the basic structure of our machine
    learning statistic methods (classification, regression, clustering ...).
    """
    
    def __init__(self, data_train, data_test, data_y_var, models_name,
        nb_folds, consumption_method) :
        """

        Attributes:
            data_train (obj): Train data, store in a pandas.DataFrame object.
            data_test (obj): test data, store in a pandas.DataFrame object.
            data_y_var (str): Column name of the response variable.
            models_name (list): List of models (similar classes) names.
            nb_folds (int): Folds numbers. 
            consumption_method (str): Consumption measurement package.
        """

        # train/test split
        self._X_train = data_train.loc[:, data_train.columns != data_y_var] # ?? for clust
        self._y_train = data_train[data_y_var]
        self._X_test = data_test.loc[:, data_test.columns != data_y_var]
        self._y_test = data_test[data_y_var]
        self._nb_folds = nb_folds
        self._consumption_method = consumption_method
        
        
        for model_name in models_name:
            module_name = model_name.lower()
            models = []
            # get the suitable method class
            try :
                module_path = "greenml.runner.methods.models." + module_name
                module = importlib.import_module(module_path)
                constr_model = getattr(module, model_name)
                models.append(constr_model(self._X_train, self._y_train, self._X_test,
                    self._nb_folds, self._consumption_method))
            except (ImportError, AttributeError):
                raise ValueError("Unknown " + module_path)
        self._models = models


    @abstractmethod
    def __compute_metrics(self, key, y_pred):
        pass


    def fit(self) :
        """This protected method aim to construct the suitable model and method. 

        Args:
            nb_folds (int, optional): Set the folds numbers in the cross validation. Defaults to 10.
        """
        return {model.model_name : model.fit_cv() for model in self.__models}


    def get_metrics(self):
        """get metrics for each models.

        Returns:
            dict : computed metrics of all fitted models. 
        """
        try: 
            models_metrics = {
                self.__compute_metrics(model.model_name, model.predict())
                    for model in self._models
            }
        except NotFittedError as e: # catch non fitted model error
            print(repr(e))
            raise
                
        return models_metrics