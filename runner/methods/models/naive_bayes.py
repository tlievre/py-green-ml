from greenml.runner.methods.models.model import Model
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV

# /!\ doesn't work, need to be fixed
class Naive_Bayes(Model):
    """Naives Bayes class. It uses the GaussianNB from sklearn.naive_bayes.
    Inherit from the abstract class model
    """

    def __init__(self, X_train, y_train, X_test, nb_folds, consumption_method,
        params = {'var_smoothing': [1e-9, 1e-7, 1e-5, 1e-3, 1e-1]}):
        """_summary_

        Args:
            X_train (pd.DataFrame): Train set predictors.
            y_train (pd.DataFrame): Train set responses.
            X_test (pd.DataFrame): Test set predictors.
            nb_folds (int): Folds numbers used in cross validation.
            params (dict, optional): Contains listes of tuning parameters given
            by sklearn GaussianNB() model. Defaults to 
                {'var_smoothing': [1e-9, 1e-7, 1e-5, 1e-3, 1e-1]}.
        """
        super().__init__(X_train, y_train, X_test, nb_folds,
            consumption_method)
        self._parameters = params
        self.__grid = GridSearchCV(GaussianNB(), params,
            cv = self._nb_folds, verbose = True)
    

    def fit_cv(self):
        """Compute the predicted response vector given by the trained model GaussianNB.

        Returns:
            float: measure.
        """
        if self._measurement is None:
            self.__grid.fit(self._X_train, self._y_train)
            return_value = None
        else :
            # training measure
            self._measurement.begin()
            self.__grid.fit(self._X_train, self._y_train)
            self._measurement.end()
            return_value = self._measurement.convert()
        return return_value

    def predict(self):
        """Compute the predicted response vector given sklearn trained model GaussianNB.

        Returns:
            array: 1-D predicted response vector.
        """
        return self.__grid.predict(self._X_test)

    @property
    def parameters(self) -> dict:
        return self.__parameters
        