"""
codecarbon measurement class
"""

import codecarbon
import io
import logging
import json

from measurement import Measure


def fibonacci(n):
    a = 0
    b = 1
     
    # Check is n is less
    # than 0
    if n < 0:
        print("Incorrect input")
         
    # Check is n is equal
    # to 0
    elif n == 0:
        return 0
       
    # Check if n is equal to 1
    elif n == 1:
        return b
    else:
        for i in range(1, n):
            c = a + b
            a = b
            b = c
        return b


class CodeCarbonMeasurement(Measure):
    def __init__(self, tracking_mode="process"):
        _logger = logging.getLogger("codecarbon_logger")
        _logger.setLevel(logging.INFO)
        self._log_string = io.StringIO()
        _channel = logging.StreamHandler(self._log_string)
        _channel.setLevel(logging.INFO)

        _logger.addHandler(_channel)

        _carbon_logger = codecarbon.output.LoggerOutput(_logger,
                                                        logging.INFO)

        self.tracker = codecarbon.EmissionsTracker(save_to_logger = True,
                                                   logging_logger = _carbon_logger,
                                                   log_level = logging.NOTSET,
                                                   tracking_mode=tracking_mode,
                                                   save_to_file = False)

    def begin(self):
        return self.tracker.start()

    def end(self):
        _ = self.tracker.stop()
        return

    def convert(self):
        consumption_dico = json.loads(self._log_string.getvalue())
        return consumption_dico['cpu_energy']
    

        
        
