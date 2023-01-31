from greenml.runner.methods.method import Method
from sklearn import metrics


class Multi_Classification(Method):
    """Multinomial classification, could use the following implemented model :
        - Support vector machine
        - Multinomial Naives Bayes
        - Logistic regression
        - K-nearest neighbors
        - Decision tree classifier
        - Random forest classifier

    Inherit from Method abstract class.
    """


    def _compute_metrics(self, y_pred):
        """Compute metrics of a multninomial classification fitted model.

        Args:
            key (str): Key of the returned dictionnary.
            y_pred (array): Predicted vector 1-D of a fitted model. 

        Returns:
            dict: In multiclassification, it appears two method for averaging
            data (from sklearn), "micro" calculate metrics globally by
            counting the total true positives, false negatives and false
            positives. "macro" Calculate metrics for each label, and find their
            unweighted mean. get_metrics compute the following metrics below :
                - accuracy
                - precision micro/macro
                - recall micro/macro
                - f1 score micro/macro
        """
        multi_clf_metrics = {
            "accuracy": metrics.accuracy_score(self._y_test, y_pred),
            "precision_micro": metrics.precision_score(self._y_test, y_pred,
                average="micro"),
            "precision_macro": metrics.precision_score(self._y_test, y_pred,
                average="macro"),
            "recall_micro": metrics.recall_score(self._y_test, y_pred,
                average="micro"),
            "recall_macro": metrics.recall_score(self._y_test, y_pred,
                average="macro"),
            "f1_micro": metrics.f1_score(self._y_test,
                y_pred, average="micro"),
            "f1_macro": metrics.f1_score(self._y_test,
                y_pred, average="macro")
        }

        return multi_clf_metrics
