# coding: utf-8

import json
import pandas as pd
import greenml.models as models
from sklearn import metrics

class ML_method :
    
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
        
    def get_metrics(self) :
            
        # to herit
        return 






def multi_classifier(X_train, y_train, X_test, y_test, model):
    """_summary_

    Args:
        X_train (_type_): _description_
        y_train (_type_): _description_
        X_test (_type_): _description_
        y_test (_type_): _description_
        model (_type_): _description_

    Returns:
        _type_: _description_
    """
    token_list = json.load(open("greenml/models.json"))
    tok = token_list["Multi_Classification"][model]
    
    clf = getattr(models,tok)
    
    y_pred = clf(X_train, y_train, X_test, y_test)
    
    acc = metrics.accuracy_score(y_test,y_pred)
    prec_micro = metrics.precision_score(y_test,y_pred,average = "micro")
    prec_macro = metrics.precision_score(y_test,y_pred,average = "macro")
    rec_micro = metrics.recall_score(y_test,y_pred,average = "micro")
    rec_macro = metrics.recall_score(y_test,y_pred,average = "macro")
    f1_micro = metrics.f1_score(y_test,y_pred,average = "micro")
    f1_macro = metrics.f1_score(y_test,y_pred,average = "macro")
    
    return {"accuracy" : acc,
            "precision_micro": prec_micro,
            "precision_macro": prec_macro,
            "recall_micro" : rec_micro,
            "recall_macro" : rec_macro,
            "f1_micro" : f1_micro,
            "f1_macro" : f1_macro}
    
    
