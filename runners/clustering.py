from sklearn.metrics.cluster import adjusted_rand_score
from greenml.runners.ml_method import ML_method

class Clustering(ML_method):
    """

    Args:
        ML_method (_type_): _description_
    """

    def get_metrics(self):
        """_summary_
        """

        # /!\ to complete
        y_pred = self.__run()

        adj_rand_score = adjusted_rand_score(self._y_test, y_pred)


        return {}
