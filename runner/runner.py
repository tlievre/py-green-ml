# coding: utf-8

import json
import pandas as pd
import importlib

# from exception import Exception

#/!\ add when measure methods will be implement
# from measure import PyRaplMeasurement, CodeCarbonMeasurement

class Runner() :
    """This abstract class implement the basic structure of our machine
    learning statistic methods (classification, regression, clustering ...).
    """
    
    def __init__(self, config) :
        """
        Args:
            config (dict): contain all parameters of a Method object.
            the dictionnary should be organized like below :
                - name : str
                - path : str
                - task : str
                - y : array
                - token : str
                - folds : int
            model_path (str): path of the machine learning model.
        """
        # config load
        # /!\ types should be test
        self._data_name = config["name"]
        self._data_path = config["path"]
        self._data_y_var = config["y"]
        self._nb_folds = config["folds"]

        # get the measure method
        if (measure := config["measure"]) == None:
            self._measure = None 
        elif measure == "pyRapl":
            self._measure = PyRaplMeasurement()
        elif measure == "codecarbon":
            self._measure = CodeCarbonMeasurement()
        else:
            raise Exception("measure value does't refer an existing \
                measurement method")
        
        # data load /!\ à modifier dans REDIS
        try:
            data_train = pd.read_csv(str(self._data_path) + str(self._data_name) \
                + "_train.csv")
            data_test = pd.read_csv(str(self._data_path) + str(self._data_name) \
            + "_test.csv")
        except FileNotFoundError:
            print("File not found.")
        except pd.errors.EmptyDataError:
            print("No data")
        except pd.errors.ParserError:
            print("Parse error")
        except Exception:
            print("Some other exception")

        # fetch suitable task and token
        # task and token to test
        task = config["task"]
        token = config["mod_token"]
        with open("greenml/models.json") as f:
            token_list = json.load(f)
        tok = token_list[task][token]
        

        # get the suitable method class
        try :
            method_path = "greenml.runner.methods." + task.lower()
            method = importlib.import_module(method_path)
            constr_method = getattr(method, task)
            self._method = constr_method(data_train, data_test, config["y"], tok, config["folds"])
        except (ImportError, AttributeError):
            raise ValueError("Unknown " + method_path)

    
    def run(self) :
        """This protected method aim to call the suitable model.

        Returns:
            array : 1-D vector of the predicted value
        """
        # train the model
        self._method.fit()
        # get the metrics
        return self._method.get_metrics()