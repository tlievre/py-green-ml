# coding: utf-8

from sklearn import metrics
from greenml.runners.ml_method import ML_method

class Multi_Classifier(ML_method):
    """Multinomial classification, could use the following implemented model :
        - Support vector machine
        - Multinomial Naives Bayes
        - Logistic regression
        - K-nearest neighbors
        - Decision tree classifier
        - Random forest classifier

    Inherit from ML_method abstract class.
    """
    
    def get_metrics(self):
        """compute multinomial classification metrics. 
        
        Returns:
            dict : In multiclassification, it appears two method for averaging
            data (from sklearn), "micro" calculate metrics globally by
            counting the total true positives, false negatives and false
            positives. "macro" Calculate metrics for each label, and find their
            unweighted mean. get_metrics compute the following metrics below :
                - accuracy
                - precision micro/macro
                - recall micro/macro
                - f1 score micro/macro
        """
        #/!\ Should we add ROC and AUC ?

        y_pred = self._run()

        acc = metrics.accuracy_score(self._y_test, y_pred)
        prec_micro = metrics.precision_score(
            self._y_test, y_pred, average="micro")
        prec_macro = metrics.precision_score(
            self._y_test, y_pred, average="macro")
        rec_micro = metrics.recall_score(
            self._y_test, y_pred, average="micro")
        rec_macro = metrics.recall_score(
            self._y_test, y_pred, average="macro")
        f1_micro = metrics.f1_score(self._y_test, y_pred, average="micro")
        f1_macro = metrics.f1_score(self._y_test, y_pred, average="macro")

        return {
            "accuracy": acc,
            "precision_micro": prec_micro,
            "precision_macro": prec_macro,
            "recall_micro": rec_micro,
            "recall_macro": rec_macro,
            "f1_micro": f1_micro,
            "f1_macro": f1_macro
        }
