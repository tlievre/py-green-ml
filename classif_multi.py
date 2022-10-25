# coding: utf-8

import json
import greenml.models as models
from sklearn import metrics

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
    prec = metrics.precision_score(y_test,y_pred)
    rec = metrics.recall_score(y_test,y_pred)
    f1 = metrics.f1_score(y_test,y_pred)
    auc = metrics.roc_auc_score(y_test,y_pred)
    
    return {"accuracy" : acc,"precision": prec,"recall" : rec,"f1" : f1,"auc": auc}
    
    
