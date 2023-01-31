from greenml.runner.methods.models.model import Model
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
        self.__parameters = params
        self.__grid = GridSearchCV(DecisionTreeClassifier(), params,
            cv = self._nb_folds, verbose = True)

    def fit_cv(self):
        """Compute the predicted response vector given by the trained model
        DecisionTreeClassifier().

        Returns:
            float: measure.
        """
        if self.__measurement is None:
            return_value = None
        else :
            # training measure
            self.__measurement.begin()
            self.__grid.fit(self._X_train, self._y_train)
            self.__measurement.end()
            return_value = self.__measurement.convert()
        return return_value

    def predict(self):
        """Compute the predicted response vector given sklearn trained model
        DecisionTreeClassifier().

        Returns:
            array: 1-D predicted response vector.
        """
        return self.__grid.predict(self._X_test)

    @property
    def parameters(self) -> dict:
        return self.__parameters