from greenml.runner.methods.models.model import Model
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV

# /!\ doesn't work, need to be fixed
class Naive_Bayes(Model):
    """Naives Bayes class. It uses the MultinomialNB from sklearn.naive_bayes.
    Inherit from the abstract class model
    """

    def __init__(self, X_train, y_train, X_test, nb_folds,
        params = {'alpha': [0.00001, 0.0001, 0.001, 0.1, 1, 10, 100, 1000]}):
        """_summary_

        Args:
            X_train (pd.DataFrame): Train set predictors.
            y_train (pd.DataFrame): Train set responses.
            X_test (pd.DataFrame): Test set predictors.
            nb_folds (int): Folds numbers used in cross validation.
            params (dict, optional): Contains listes of tuning parameters given
            by sklearn MultinomialNB() model. Defaults to 
                {'alpha': [0.00001, 0.0001, 0.001, 0.1, 1, 10, 100, 1000]}.
        """
        super().__init__(X_train, y_train, X_test, nb_folds)
        self._parameters = params
        self.__grid = GridSearchCV(MultinomialNB(), params,
            cv = self._nb_folds, verbose = True)
    

    def fit_cv(self):
        """Compute the predicted response vector given by the trained model MultinomialNB.

        Returns:
            array: 1-D predicted repsonse vector.
        """
        
        self.__grid.fit(self._X_train, self._y_train)

    def predict(self):
        """Compute the predicted response vector given sklearn trained model MultinomialNB.

        Returns:
            array: 1-D predicted response vector.
        """
        return self.__grid.predict(self._X_test)

    @property
    def parameters(self) -> dict:
        return self.__parameters