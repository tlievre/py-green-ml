from scipy.special import comb
import pandas as pd

from sklearn.metrics import fowlkes_mallows_score
from sklearn.metrics.cluster import adjusted_rand_score, rand_score, homogeneity_score
from greenml.runner.methods.method import Method


class Clustering(Method):
    """Clustering class, could use the following implemented model :
        - k_means
    
    Inherit from Method abstract class.
    """

    def __rand_index(self, gold_standards, clusters):
        """Rand index function. It allows to retrieve the confusion matrix of
        the rand index binary classifier.

        Args:
            gold_standards (array): Gold standards labels in our case category column. Defaults to df.category[:500].
            clusters (array): Predictions provide by the clustering algorithm. Defaults to clustering[:500].

        Returns:
            dict: a dictionnary with the following keys 'rand_index' and 'confusion_matrix'.
        """

        # compute the rand index data frame
        df_rand_index = pd.DataFrame({"gold_standard" : gold_standards, "cluster" : clusters})
        
        # compute scores of the positive gold standard (tp/a + fn/b)
        positive_gd = comb(df_rand_index.gold_standard.value_counts().values, 2).sum()

        # compute scores of the positive gold standard (fn/c + tn/d)
        positive_pred = comb(df_rand_index.cluster.value_counts().values, 2).sum()

        # compute combiation of all couples with same clusters given same gold standard
        a = sum(comb(df_rand_index[df_rand_index.gold_standard == gs].cluster.value_counts().values, 2).sum()
            for gs in set(df_rand_index.gold_standard))

        # extract the other metrics of the confusion matrix
        b = positive_gd - a
        c = positive_pred - a
        d = comb(df_rand_index.shape[0], 2) - a - b - c
        
        # compute the rand index (accuracy score of the binary classifier)
        ri = (a + d) / (a + b + c + d)

        return {
            'rand_index' : ri,
            'confusion_matrix' : {'a' : a, 'b' : b, 'c' : c, 'd' : d},
            'rand_distance' : 1 - ri
        }
    
    def _compute_metrics(self, y_pred):
        """Compute metrics of a clustering fitted model.

        Args:
            key (str): Key of the returned dictionnary.
            y_pred (array): Predicted vector 1-D of a fitted model.

        Returns:
            dict: It contains the following metrics (implement by sklearn) :
                - adjusted rand index
                - rand index
                - recall
                - precision
                - F1-score
                - homogeneity score
                - fowlkes mallows index
        """
        # compute the confusion matrix of the rand classifier
        rand_conf = self.__rand_index(self._y_test, y_pred)
        rand_conf = rand_conf['confusion_matrix']

        # compute recall and precision from the rand classifier confusion matrix
        recall = rand_conf['a'] / (rand_conf['a'] + rand_conf['b'])
        precision = rand_conf['a'] / (rand_conf['a'] + rand_conf['c'])

        metrics = {
            'adj_rand_index' : adjusted_rand_score(self._y_test, y_pred),
            'rand_index' : rand_score(self._y_test, y_pred),
            'recall' : recall,
            'precision' : precision,
            'f1' : 2 * recall * precision / (recall + precision), 
            'h_score' : homogeneity_score(self._y_test, y_pred),
            'fowlkes_mallows_index' : fowlkes_mallows_score(self._y_test, y_pred)
        }
        return metrics
