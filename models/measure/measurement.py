"""
Measurement abstract class
"""


from abc import ABC, abstractmethod

class Measure(ABC):
    @abstractmethod
    def begin(self):
        pass

    @abstractmethod
    def end(self):
        pass

    @abstractmethod
    def convert(self, kwh_cst=2.778e-7):
        pass
