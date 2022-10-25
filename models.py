from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
import numpy as np

import warnings
warnings.filterwarnings("ignore")

def SVM_Linear(X_train,y_train,X_test,y_test):
    """_summary_

    Args:
        X_train (_type_): _description_
        y_train (_type_): _description_
        X_test (_type_): _description_
        y_test (_type_): _description_

    Returns:
        _type_: _description_
    """
    from sklearn.svm import LinearSVC
    
    clf = LinearSVC()
    
    hparam = {
        "tol" : np.linspace(1e-5,1e-2,5),
        "C" : np.linspace(1e-4,1,5)
    }
    
    grid = GridSearchCV(clf, hparam, cv=10,verbose=True)

    grid.fit(X_train, y_train)
    
    y_pred = grid.best_estimator_.predict(X_test)
    
    return y_pred