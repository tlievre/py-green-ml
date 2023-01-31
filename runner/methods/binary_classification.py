from greenml.runner.methods.method import Method

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

    def __compute_metrics(self, key, y_pred):
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
            "accuracy": metrics.accuracy_score(self._y_test, y_pred),
            "precision": metrics.precision_score(self._y_test, y_pred),
            "recall": metrics.recall_score(self._y_test, y_pred),
            "f1": metrics.f1_score(self._y_test, y_pred)
        }
        return {key : metrics}
