from sklearn import metrics
from sklearn.exceptions import NotFittedError
from greenml.runner.methods.ml_method import ML_method
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error

class Regression(ML_method):
    """Clustering class, could use the following implemented model :
        - Linear Regression

    Inherit from ML_method abstract class.
    """

    def get_metrics(self):
        """Compute some regression metrics. Should be trained before

        Returns:
            dict: It contains the sklearn metrics below :
                - R²
                - explained variance (similar to R²)
                - Minimum square error
        """
        
        try: 
            y_pred = self._model.predict()
        except NotFittedError as e: # catch non fitted model error
            print(repr(e))
            raise

        return {
            'r2_score' : r2_score(self._y_test, y_pred),
            'exp_var' : explained_variance_score(self._y_test, y_pred),
            'mse' : mean_squared_error(self._y_test, y_pred)
        }
