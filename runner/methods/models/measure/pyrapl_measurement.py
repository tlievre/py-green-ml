"""
pyrapl Measurement class
"""

import pyRAPL

#from greenml.runner.methods.models.measure.measurement import Measure

from measurement import Measure

class PyRAPLMeasurement(Measure):
    def __init__(self):
        pyRAPL.setup()
        self.output = pyRAPL.outputs.DataFrameOutput()
        self.measure = pyRAPL.Measurement("test",output=self.output)

    def begin(self):
        """Begins measurement with pyRAPL

        :returns:
        :rtype: 

        """
        return self.measure.begin()

    def end(self):
        """Ends measurement with pyRAPL

        :returns: 
        :rtype: 

        """
        return self.measure.end()
 
    
    def convert(self, kwh_cst=2.778e-7):
        """converts micro joules to KWh

        :param float kwh_cst: value of 1J in KWh
        :returns: consumption in KWh
        :rtype: float

        """
        measure_uj = self.measure.result.pkg[0]
        measure_j = self.measure.result.pkg[0]*(10**-6)
        return kwh_cst*measure_j
