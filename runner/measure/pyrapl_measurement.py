"""
pyrapl Measurement class
"""

import pyRAPL

from measurement import Measurement
from functools import wraps


## decorator not finished
## make sure to launch the model
def measure(func):
    """ decorator measuring the execution time
    of its argument

    :parameter func: function to measure the consumption of
    :returns: result of measurement
    :rtype: 

    """
    @wraps(func)
    def consumption_wrapper():
##        model, measurement = func()
        if measurement:
            pyrapl_obj = PyRAPLMeasurement(measurement_output="csv")
            return pyrapl_obj.measurement(model)
        return model()
    return wrapper


class PyRAPLMeasurement(Measurement):
    def __init__(self, measurement_output="csv",
                 output_file_name="pyrapl-result.csv"):
        if measurement_output == "csv":
            self.output = pyRAPL.outputs.CSVOutput(output_file_name)
        elif measurement_output == "stdout":
            self.output = pyRAPL.outputs.PrintOutput(raw=False)
        elif measurement_output == "data_frame":
            self.output = pyRAPL.outputs.DataFrameOutput()
        else:
            raise ValueError(f"{measurement_output} not supported")

    def measurement(self, model):
        pyRAPL.setup()
        measure = pyRAPL.Measurement(output=self.output)
        measure.begin()
        model()
        measure.end()
        if isinstance(self.output, pyRAPL.outputs.DataFrameOutput):
            data = self.output.data
            return data
        return True
        
