# coding: utf-8

import json
import pandas as pd
import numpy as np
import importlib

from numpy.random import randint
from greenml.runner.exception import EmptyModelsNameList

#/!\ add when measure methods will be implement
# from measure import PyRaplMeasurement, CodeCarbonMeasurement

class Runner() :
    """This abstract class implement the basic structure of our machine
    learning statistic methods (classification, regression, clustering ...).
    """
    
    def __init__(self, config, draw_rate = 2/3) :
        """
        Attributes:
            config (dict): contain all parameters of a Method object.
            the dictionnary should be organized like below :
                - name : str
                - path : str
                - task : str
                - y : array
                - token : list
                - folds : int
                - measure : str
            model_path (str): path of the machine learning model.
        """
        # config loading
        self._data_name = config["name"]
        self._data_path = config["path"]
        self._data_y_var = config["y"]
        self._nb_folds = config["folds"]
        self._consumption_method = config["measure"]
        
        
        # data load
        try:
            data_train = pd.read_csv(str(self._data_path) + str(self._data_name) \
                + "_train.csv")
            data_test = pd.read_csv(str(self._data_path) + str(self._data_name) \
                + "_test.csv")
        except FileNotFoundError:
            print("File not found.")
            raise
        except pd.errors.EmptyDataError:
            print("No data")
            raise
        except pd.errors.ParserError:
            print("Parse error")
            raise
        except Exception:
            print("Some other exception")
            raise

        # random sample
        max_train = len(data_train)
        self.__train_lines = randint(round((draw_rate)*max_train), max_train)
        data_train = data_train.sample(n = self.__train_lines)

        random_rate = self.__train_lines / max_train

        self.__test_lines = round(random_rate * len(data_test))
        data_test = data_test.sample(n = self.__test_lines)
        
        # fetch suitable task and token
        # task and token to test
        task = config["task"]
        mod_tokens = config["mod_tokens"]

        # read the models.json file
        with open("greenml/models.json") as f:
            token_list = json.load(f)

        try:
            models_name = [model for mod_token,model in token_list[task].items() if mod_token in mod_tokens]
        except KeyError as err:
            print("{} isn't refer in the dictionnary".format(err))
            raise
        
        # check empty list
        if not models_name:
            raise EmptyModelsNameList("models_name is currently empty") 

        # get the suitable method class
        try :
            method_path = "greenml.runner.methods." + task.lower()
            method = importlib.import_module(method_path)
            constr_method = getattr(method, task)
            self._method = constr_method(data_train, data_test, config["y"],
                models_name, config["folds"], config["measure"])
        except (ImportError, AttributeError):
            raise ValueError("Unknown " + method_path)

    
    def run(self) :
        """This protected method aim to call the suitable model.

        Returns:
            array : 1-D vector of the predicted value
        """
        # train the model
        measurement = self._method.fit()
        # get the metrics
        return {
            'measurement' : measurement,
            'metrics' : self._method.get_metrics(),
            'nb_lines_train' : self.__train_lines,
            'nb_lines_test' : self.__test_lines
        }