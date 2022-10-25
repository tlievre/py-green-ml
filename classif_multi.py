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
    
    
