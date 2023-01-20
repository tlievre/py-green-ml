from sklearn import metrics
from greenml.runners.ml_method import ML_method
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error

class Regressor(ML_method):
    """

    Args:
        ML_method (_type_): _description_
    """

    def get_metrics(self):
        """_summary_
        """
        
        y_pred = self.__run()

        return {
            'r2_score' : r2_score(self._y_test, y_pred),
            'exp_var' : explained_variance_score(self._y_test, y_pred),
            'mse' : mean_squared_error(self._y_test, y_pred)
        }
