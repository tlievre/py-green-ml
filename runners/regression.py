from sklearn import metrics
from greenml.runners.ml_method import ML_method
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error

class Regression(ML_method):
    """Clustering class, could use the following implemented model :
        - Linear Regression

    Inherit from ML_method abstract class.
    """

    def get_metrics(self):
        """Compute some regression metrics.

        Returns:
            dict: It contains the sklearn metrics below :
                - R²
                - explained variance (similar to R²)
                - Minimum square error
        """
        
        y_pred = self._run()

        return {
            'r2_score' : r2_score(self._y_test, y_pred),
            'exp_var' : explained_variance_score(self._y_test, y_pred),
            'mse' : mean_squared_error(self._y_test, y_pred)
        }
