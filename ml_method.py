# coding: utf-8

import json
import pandas as pd
import greenml.models as models
from sklearn import metrics

class ML_method :
    """_summary_
    """
    
    def __init__(self,config) :
        self.data_name = config["name"][0]
        self.data_path = config["path"][0]
        self.task = config["task"][0]
        self.data_y_var = config["y"][0]
        self.token = config["mod_token"][0]
        
        data_train = pd.read_csv(str(self.data_path) + str(self.data_name) + "_train.csv")
        data_test = pd.read_csv(str(self.data_path) + str(self.data_name) + "_test.csv")

        X_train = data_train.loc[:, data_train.columns != self.data_y_var] # ?? for clust
        y_train = data_train[self.data_y_var]
        X_test = data_test.loc[:, data_test.columns != self.data_y_var]
        y_test = data_test[self.data_y_var]
        
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
        
        self.model = None
        
        self.y_pred = None
        
        self.metrics = {}
        
    def run(self) :
        
        token_list = json.load(open("greenml/models.json")) # voir si ça marche 
        tok = token_list[self.task][self.token]
        
        self.model = getattr(models,tok)
        
        self.y_pred = self.model(self.X_train, self.y_train, self.X_test, self.y_test)
        