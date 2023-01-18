from sklearn import metrics
from greenml.runners.ml_method import ML_method

class Binar_Classifier(ML_method):
    """

    Args:
        ML_method (_type_): _description_
    """

    def get_metrics(self):
        """_summary_
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
