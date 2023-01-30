import numpy as np

from greenml.runner.methods.models.model import Model

from sklearn.model_selection import GridSearchCV
from tensorflow import keras

from scikeras.wrappers import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping



class Perceptron(Model):
    """Linear support vector machine class. It uses LinearSVC() from
    sklearn.svm. Inherit from Model abstract class.
    """

    # params need to be test
    def __init__(self, X_train, y_train, X_test, nb_folds ,
        params = {
            'optimizer__learning_rate': [1e-3,1e-4,1e-5],
            'batch_size': [8,16,32],
            'epochs' :[100]
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
                    'learning_rate': [1e-3,1e-4,1e-5],
                    'batch_size': [8,16,32],
                    'epochs' :[100]
                }.
        """
        super().__init__(X_train, y_train, X_test, nb_folds)

        self.__parameters = params
        
        n_classes = len(np.unique(self._y_train))
        if(n_classes == 2):
            loss = 'binary_crossentropy'
            act = 'sigmoid'
        
        else:
            loss = 'categorical_crossentropy'
            act = 'softmax'
            
        def build_net():
            """
            """
            
            net = Sequential()
            net.add(Dense(n_classes,input_shape=(np.shape(X_train)[1],),activation=act))
            return net
        
        early_stop = EarlyStopping("val_loss",patience=5)
        
        self._Net = KerasClassifier(model=build_net,
                                    verbose=1,
                                    callbacks=early_stop,
                                    optimizer= "adam",
                                    loss=loss)
        
        self._grid = GridSearchCV(estimator = self._Net, param_grid = params, cv = self._nb_folds, n_jobs=1,verbose = True)



    def fit_cv(self):
        """.
        """
        
        self._grid.fit(self._X_train, self._y_train)
        
        
        
        
    
    def predict(self):
        """Compute the predicted response vector 
        """
        return self._grid.predict(self._X_test)

    @property
    def parameters(self) -> dict:
        return self.__parameters