from sklearn import metrics
from greenml.runners.ml_method import ML_method

class Binar_Classifier(ML_method):
    """Binary classification, could use the following implemented model :
        - Support vector machine
        - Multinomial Naives Bayes
        - Logistic regression
        - K-nearest neighbors
        - Decision tree classifier
        - Random forest classifier
    
    Args:
        ML_method (class): Ineherited abstract class.
    """

    def get_metrics(self):
        """compute the binary classification metrics.

        Returns :
            dict : It gives the metrics below :
                - accuracy
                - precision
                - recall
                - F1-score
        """
        y_pred = self._run()

        acc = metrics.accuracy_score(self._y_test, y_pred)
        prec = metrics.precision_score(self._y_test, y_pred)
        rec = metrics.recall_score(self._y_test, y_pred)
        f1 = metrics.f1_score(self._y_test, y_pred)

        return {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1
        }
