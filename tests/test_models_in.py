import sys
import pandas as pd
import importlib
import os

sys.path.append("../../")

l = list(filter(lambda x: ".py" in x, os.listdir("../models")))

def test_init(model):

    X = pd.DataFrame([[1, 4], [2, 4], [3, 6]], columns=["A", "B"])
    X_t = pd.DataFrame([[3, 4], [4, 8]], columns=["A", "B"])
    y = pd.Series([1, 0, 1])

    M = model(X, y, X_t)
    
def test_fit_CV(model):

    X = pd.DataFrame([[1, 4], [2, 4], [3, 6]], columns=["A", "B"])
    X_t = pd.DataFrame([[3, 4], [4, 8]], columns=["A", "B"])
    y = pd.Series([1, 0, 1])

    M = model(X, y, X_t)
    M.fit_CV()

for file in l:

    tok = file.split(".")[0]

    models_file = importlib.import_module("models."+tok)

    model = getattr(models_file, tok.capitalize())
    
    test_init(model)
    test_fit_CV(model) 
    
