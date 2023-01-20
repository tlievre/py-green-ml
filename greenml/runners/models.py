from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore")


class Model:

    def __init__(self, X_train, y_train, X_test):

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test

        self.hparam = {}

        self.method = None

        self.y_pred = None

    def fit_CV(self, nb_fold=10):

        grid = GridSearchCV(self.method, self.hparam, cv=nb_fold, verbose=True)

        grid.fit(self.X_train, self.y_train)

        self.y_pred = grid.best_estimator_.predict(self.X_test)

        return self.y_pred
