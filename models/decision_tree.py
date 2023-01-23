from greenml.models.model import Model
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

class Decision_Tree(Model):

    def __init__(self, X_train, y_train, X_test, nb_folds,
        params = {
            'criterion': ["gini", "entropy", "log_loss"],
            'splitter': ["best", "random"],
            'max_features' : ["auto", "sqrt", "log2"]}
        ):
        """
        Args:
            X_train (pd.DataFrame): Train set predictors.
            y_train (pd.DataFrame): Train set responses.
            X_test (pd.DataFrame): Test set predictors.
            nb_folds (int): Folds numbers used in cross validation.
            params (dict, optional): Contains listes of tuning parameters given
            by sklearn DecisionTreeClassifier() model. Defaults to {'criterion': ["gini", "entropy", "log_loss"], 'splitter': ["best", "random"], 'max_features' : ["auto", "sqrt", "log2"]}.
        """
        super().__init__(X_train, y_train, X_test, nb_folds)
        self._hparam = params

    def _fit_cv(self):
        """Compute the predicted response vector given by the trained model
        DecisionTreeClassifier().

        Returns:
            array: 1-D predicted repsonse vector.
        """
        grid = GridSearchCV(DecisionTreeClassifier(), self._hparam,
            cv = self._nb_folds, verbose = True)
        grid.fit(self._X_train, self._y_train)
        return grid