"""
SCRIPT EXAMPLE FOR EXECUTOR :

Run machine learning algorithm to a selected dataset for a given task (Binary Classification,
Multi-class Classification, Regression, Clustering)

Input : 
    config : json file, config file with token :
        name : str, dataset name
        path : str, path of the directory containing [name]_train.csv and [name]_test.csv
        task : str, one in {Binary Classification, Multi-class Classification, Regression, Clustering}
        y : str, name of response for supervised learning
        
Output : 
    stdout : results (metrics for the given task)
"""

import sys
import json
import pyRAPL

from greenml.runners.classif_multi import Multi_Classifier
from codecarbon import EmissionsTracker
tracker = EmissionsTracker()

#pyRAPL.setup()
#csv_output = pyRAPL.outputs.CSVOutput("results.csv")

# config = json.load(open(config.json))
config = json.load(open(sys.argv[1]))

# Call Task function with X_train, y_train, X_test, y_test + model

clf = Multi_Classifier(config)

#meter = pyRAPL.Measurement('clf.run', output=csv_output)
#meter.begin()
tracker.start()
clf.run(nb_fold=2)
clf.get_metrics()
tracker.stop()
#meter.end()
#csv_output.save()
# print results on stdout

print(clf.metrics)
