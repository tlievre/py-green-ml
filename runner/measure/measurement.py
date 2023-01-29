"""
Measurement abstract class
"""


from abc import ABC, abstractmethod

class Measurement(ABC):
    @abstractmethod
    def __init__(self, measurement_output="csv"):
        self.measurement_output = measurement_output
    
    @abstractmethod
    def measurement(self, fun):
        pass
