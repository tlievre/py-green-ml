from greenml.runner.methods.method import Method
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error

class Regression(Method):
    """Clustering class, could use the following implemented model :
        - Linear Regression

    Inherit from Method abstract class.
    """

    def __compute_metrics(self, key, y_pred):
        """Compute metrics of a regression fitted model.

        Args:
            key (str): Key of the returned dictionnary.
            y_pred (array): Predicted vector 1-D of a fitted model. 

        Returns:
            dict: It contains the sklearn metrics below :
                - R²
                - explained variance (similar to R²)
                - Minimum square error
        """
        
        metrics = {
            'r2_score' : r2_score(self._y_test, y_pred),
            'exp_var' : explained_variance_score(self._y_test, y_pred),
            'mse' : mean_squared_error(self._y_test, y_pred)
        }

        return {key : metrics}
