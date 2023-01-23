import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV

from greenml.models.model import Model

class SVM_Linear(Model):
    """Linear support vector machine class. It uses LinearSVC() from
    sklearn.svm. Inherit from Model abstract class.
    """

    # params need to be test
    def __init__(self, X_train, y_train, X_test, nb_folds ,
        params = {
            'tol': np.linspace(1e-5, 1e-2, 5),
            'C': np.linspace(1e-3, 1, 5)
        }):
        """
        Args:
            X_train (pd.DataFrame): Train set predictors.
            y_train (pd.DataFrame): Train set responses.
            X_test (pd.DataFrame): Test set predictors.
            nb_folds (int): Folds numbers used in cross validation.
            params (dict, optional): Contains listes of tuning parameters given
            by sklearn LinearSVC() model. Defaults to
                {
                    'tol': np.linspace(1e-5, 1e-2, 5),
                    'C': np.linspace(1e-3, 1, 5)
                }.
        """
        super().__init__(X_train, y_train, X_test, nb_folds)

        self._hparam = params

    def _fit_cv(self):
        """Compute the predicted response vector given by the trained model
        LinearSVC().

        Returns:
            array: 1-D predicted repsonse vector.
        """
        grid = GridSearchCV(LinearSVC(), self._hparam, cv = self._nb_folds,
            verbose = True)
        grid.fit(self._X_train, self._y_train)
        return grid