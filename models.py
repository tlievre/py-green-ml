from sklearn.model_selection import GridSearchCV
import numpy as np

import warnings
warnings.filterwarnings("ignore")

def SVM_Linear(X_train,y_train,X_test,nb_fold=10):
    """ Apply Cross validation on C and tol hparameters of LinearSVC from scikit-learn
    and make a prediction with the best hparameters

    Args:
        X_train (pandas Dataframe): train set 
        y_train (pandas Series): train labels
        X_test (pandas Dataframe): test set
        nb_fold (int) : number of fold for cross validation

    Returns:
        array: predictions of test labels
    """
    from sklearn.svm import LinearSVC
    
    clf = LinearSVC()
    
    hparam = {
        "tol" : np.linspace(1e-5,1e-2,5),
        "C" : np.linspace(1e-4,1,5)
    }
    
    grid = GridSearchCV(clf, hparam, cv=nb_fold,verbose=True)

    grid.fit(X_train, y_train)
    
    y_pred = grid.best_estimator_.predict(X_test)
    
    return y_pred