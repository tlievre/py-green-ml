from greenml.models.model import Model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV


class Linear_Regression(Model):
    """Linear regression machine class. It uses LinearRegression() from
    sklearn.svm. Inherit from Model abstract class.
    """

    def __init__(self, X_train, y_train, X_test, nb_folds,
        params = {
            'positive' : [True, False],
            'fit_intercept' : [True, False]} # not necessary
        ):
        """
        Args:
            X_train (pd.DataFrame): Train set predictors.
            y_train (pd.DataFrame): Train set responses.
            X_test (pd.DataFrame): Test set predictors.
            nb_folds (int): Folds numbers used in cross validation.
            params (dict, optional): Contains listes of tuning parameters given
            by sklearn LinearRegression() model. Defaults to
                {
                    'positive' : [True, False],
                    'fit_intercept' : [True, False]
                }.
        """
        super().__init__(X_train, y_train, X_test, nb_folds)
        self._hparam = params

    def _fit_cv(self):
        """Compute the predicted response vector given by the trained model
        LinearRegression().

        Returns:
            array: 1-D predicted repsonse vector.
        """
        grid = GridSearchCV(LinearRegression(), self._hparam, cv = self._nb_folds, verbose = True)
        grid.fit(self._X_train, self._y_train)
        return grid