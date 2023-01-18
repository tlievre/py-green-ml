# coding: utf-8

import json
import pandas as pd
import importlib

from abc import ABC, abstractmethod


class ML_method(ABC) :
    """_summary_
    """
    
    def __init__(self,config) :
        """_summary_

        Args:
            config (_type_): _description_
        """
        # config load
        self._data_name = config["name"]
        self._data_path = config["path"]
        self._task = config["task"]
        self._data_y_var = config["y"]
        self._token = config["mod_token"]
        self._nb_folds = config["folds"]
        
        # data load /!\ à modifier dans REDIS
        data_train = pd.read_csv(str(self._data_path) + str(self._data_name) + "_train.csv")
        data_test = pd.read_csv(str(self._data_path) + str(self._data_name) + "_test.csv")

        # train/test split
        self._X_train = data_train.loc[:, data_train.columns != self._data_y_var] # ?? for clust
        self._y_train = data_train[self._data_y_var]
        self._X_test = data_test.loc[:, data_test.columns != self._data_y_var]
        self._y_test = data_test[self._data_y_var]

    
    def _run(self) :
        """_summary_

        Args:
            nb_folds (int, optional): _description_. Defaults to 10.
        """
        
        # fetch suitable task and token
        token_list = json.load(open("greenml/models.json")) # /!\ To fix LORYS !!
        tok = token_list[self._task][self._token]
        tok2 = tok.lower()

        # get the suitable class 
        models = importlib.import_module("greenml.models." + tok2)
        constr_model = getattr(models,tok)
        model = constr_model(self._X_train, self._y_train, self._X_test,self._nb_folds)
        
        return model.predict()

    @abstractmethod
    def get_metrics(self):
        pass