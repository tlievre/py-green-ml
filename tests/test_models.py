import sys
import pandas as pd
import pytest

sys.path.append("../")

from runners.models import Model


class TestModel:
    
    @pytest.fixture
    def my_model():
        """Returns a regular Model instance"""
        X = pd.DataFrame([[1,4],[2,4],[3,6]], columns = ["A", "B"])
        X_test = pd.DataFrame([[3,4],[4,8]], columns = ["A", "B"])
        y = pd.Series([1,0,1])

        return Model(X, y, X_test)

        
    def test_fit_CV(my_model):
        """Returns a regular Model instance
        """
        assert True
    