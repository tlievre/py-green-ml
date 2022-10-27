# coding: utf-8

from sklearn import metrics
from greenml.ml_method import ML_method       


class Multi_Classifier(ML_method):
    """

    Args:
        ML_method (_type_): _description_
    """
    
    def get_metrics(self) :
        """_summary_
        """
                    
        acc = metrics.accuracy_score(self.y_test,self.y_pred)
        prec_micro = metrics.precision_score(self.y_test,self.y_pred,average = "micro")
        prec_macro = metrics.precision_score(self.y_test,self.y_pred,average = "macro")
        rec_micro = metrics.recall_score(self.y_test,self.y_pred,average = "micro")
        rec_macro = metrics.recall_score(self.y_test,self.y_pred,average = "macro")
        f1_micro = metrics.f1_score(self.y_test,self.y_pred,average = "micro")
        f1_macro = metrics.f1_score(self.y_test,self.y_pred,average = "macro")
        
        self.metrics= {"accuracy" : acc,
            "precision_micro": prec_micro,
            "precision_macro": prec_macro,
            "recall_micro" : rec_micro,
            "recall_macro" : rec_macro,
            "f1_micro" : f1_micro,
            "f1_macro" : f1_macro}


    
    
