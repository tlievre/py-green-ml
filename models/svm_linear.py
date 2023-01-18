import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV

from models import Model

class SVM_Linear(Model):

    # params need to be test
    def __init__(self, X_train, y_train, X_test, params = {'tol': np.linspace(1e-5, 1e-2, 5), 'C': np.linspace(1e-3, 1, 5)}):

        super().__init__(X_train, y_train, X_test)

        self.__hparam = params

    def __fit_cv(self):
        grid = GridSearchCV(LinearSVC(), self.__hparam, cv = self.__nb_fold, verbose = True)
        grid.fit(self.__X_train, self.__y_train)
        return grid

    def predict(self):
        grid = self.__fit_cv()
        return grid.best_estimator_.predict(self.__X_test)