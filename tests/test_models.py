import sys
import pandas as pd

sys.path.append("../")

from runners.models import Model

def test_init():
    
    X = pd.DataFrame([[1,4],[2,4],[3,6]],columns=["A","B"])
    X_t = pd.DataFrame([[3,4],[4,8]],columns=["A","B"])
    y = pd.Series([1,0,1])
    
    M = Model(X,y,X_t)
    
def test_fit_CV():
    
    X = pd.DataFrame([[1,4],[2,4],[3,6]],columns=["A","B"])
    X_t = pd.DataFrame([[3,4],[4,8]],columns=["A","B"])
    y = pd.Series([1,0,1])
    
    M = Model(X,y,X_t)
    M.fit_CV()
    
    