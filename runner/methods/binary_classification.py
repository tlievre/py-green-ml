from greenml.runner.methods.method import Method
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class Binary_Classification(Method):
    """Binary classification, could use the following implemented model :
        - Support vector machine
        - Multinomial Naives Bayes
        - Logistic regression
        - K-nearest neighbors
        - Decision tree classifier
        - Random forest classifier
    
    Inherit from Method abstract class.
    """

    def _compute_metrics(self, y_pred):
        """compute metrics of a binary classification fitted model.

        Args:
            key (str): Key of the returned dictionnary.
            y_pred (array): Predicted vector 1-D of a fitted model. 

        Returns :
            dict : It gives the metrics below :
                - accuracy
                - precision
                - recall
                - F1-score
        """
        metrics = {
            "accuracy": accuracy_score(self._y_test, y_pred),
            "precision": precision_score(self._y_test, y_pred),
            "recall": recall_score(self._y_test, y_pred),
            "f1": f1_score(self._y_test, y_pred)
        }
        return metrics
