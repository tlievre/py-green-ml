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
        self.__data_name = config["name"]
        self.__data_path = config["path"]
        self.__task = config["task"]
        self.__data_y_var = config["y"]
        self.__token = config["mod_token"]
        self.__nb_folds = config["folds"]
        
        # data load /!\ à modifier dans REDIS
        data_train = pd.read_csv(str(self.__data_path) + str(self.__data_name) + "_train.csv")
        data_test = pd.read_csv(str(self.__data_path) + str(self.__data_name) + "_test.csv")

        # train/test split
        self.__X_train = data_train.loc[:, data_train.columns != self.__data_y_var] # ?? for clust
        self.__y_train = data_train[self.__data_y_var]
        self.__X_test = data_test.loc[:, data_test.columns != self.__data_y_var]
        self.__y_test = data_test[self.__data_y_var]

    
    def __run(self) :
        """_summary_

        Args:
            nb_folds (int, optional): _description_. Defaults to 10.
        """
        
        # fetch suitable task and token
        token_list = json.load(open("greenml/models.json")) # /!\ To fix LORYS !!
        tok = token_list[self.__task][self.__token]
        tok2 = tok.lower()

        # get the suitable class 
        models = importlib.import_module("greenml.models." + tok2)
        constr_model = getattr(models,tok)
        model = constr_model(self.__X_train, self.__y_train, self.__X_test)
        
        return model.predict(self.__nb_folds)

    @abstractmethod
    def get_metrics(self):
        pass