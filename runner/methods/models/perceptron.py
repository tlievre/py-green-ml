import numpy as np

from greenml.runner.methods.models.model import Model

from sklearn.model_selection import GridSearchCV
from tensorflow import keras

from scikeras.wrappers import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping


class Perceptron(Model):
    """Perceptron class. It uses KerasClassifier from
    scikeras.wrapper. Inherit from Model abstract class.
    """

    # params need to be test
    def __init__(self, X_train, y_train, X_test, nb_folds, consumption_method,
                 params={
                     'optimizer__learning_rate': [1e-3, 1e-4, 1e-5],
                     'batch_size': [8, 16, 32],
                     'epochs': [100]
                 }):
        """
        Args:
            X_train (pd.DataFrame): Train set predictors.
            y_train (pd.DataFrame): Train set responses.
            X_test (pd.DataFrame): Test set predictors.
            nb_folds (int): Folds numbers used in cross validation.
            params (dict, optional): Contains listes of tuning parameters. 
                Defaults to
                {
                    'optimizer__learning_rate': [1e-3,1e-4,1e-5],
                    'batch_size': [8,16,32],
                    'epochs' :[100]
                }.
        """
        super().__init__(X_train, y_train, X_test, nb_folds, consumption_method)

        self.__parameters = params

        self._input_shape = (np.shape(X_train)[1],)

        self._n_classes = len(np.unique(self._y_train))
        if (self._n_classes == 2):
            self._loss = 'binary_crossentropy'
            self._act = 'sigmoid'

        else:
            self._loss = 'categorical_crossentropy'
            self._act = 'softmax'

        early_stop = EarlyStopping("val_loss", patience=5)

        def build_net():
            """ Build the network. Will be call by KerasClassifier.
            """

            net = Sequential()
            net.add(Dense(self._n_classes,
                    input_shape=self._input_shape, activation=self._act))
            return net

        self._Net = KerasClassifier(model=build_net,
                                    verbose=1,
                                    callbacks=early_stop,
                                    optimizer="adam",
                                    loss=self._loss,
                                    fit__validation_split=0.2)

        self._grid = GridSearchCV(
            estimator=self._Net, param_grid=params, cv=self._nb_folds, n_jobs=1, verbose=True)

    def fit_cv(self):
        """.Compute the training and the cross validation of the model Perceptron.
        """

        self._grid.fit(self._X_train, self._y_train)

    def predict(self):
        """Compute the predicted response vector 
        """
        return self._grid.predict(self._X_test)

    @property
    def parameters(self) -> dict:
        return self.__parameters
