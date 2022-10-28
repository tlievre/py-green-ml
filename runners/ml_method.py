# coding: utf-8

import json
import pandas as pd
import Projet_tut_test.greenml.runners.models as models

class ML_method :
    """_summary_
    """
    
    def __init__(self,config) :
        """_summary_

        Args:
            config (_type_): _description_
        """
        # config load
        self.data_name = config["name"][0]
        self.data_path = config["path"][0]
        self.task = config["task"][0]
        self.data_y_var = config["y"][0]
        self.token = config["mod_token"][0]
        
        # data load
        data_train = pd.read_csv(str(self.data_path) + str(self.data_name) + "_train.csv")
        data_test = pd.read_csv(str(self.data_path) + str(self.data_name) + "_test.csv")

        # train/test split
        X_train = data_train.loc[:, data_train.columns != self.data_y_var] # ?? for clust
        y_train = data_train[self.data_y_var]
        X_test = data_test.loc[:, data_test.columns != self.data_y_var]
        y_test = data_test[self.data_y_var]
        
        # generate attributes
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test   
        
        self.model = None
        self.y_pred = None
        self.metrics = {}
        
    def run(self,nb_fold=10) :
        """_summary_

        Args:
            nb_fold (int, optional): _description_. Defaults to 10.
        """
        
        token_list = json.load(open("greenml/models.json")) # voir si ça marche 
        tok = token_list[self.task][self.token]
        
        self.model = getattr(models,tok)
        
        clf = self.model(self.X_train, self.y_train, self.X_test)
        
        self.y_pred = clf.fit_CV(nb_fold)
        