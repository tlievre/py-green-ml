"""
Regression script
"""

import json
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# get fields of json file

with open('config.json') as f:
    content = f.read()
fields = json.loads(content)
dataset_train = fields['Path'] + '/' + fields['Name'] + '_train'  + '.csv'
dataset_test =  fields['Path'] + '/' + fields['Name'] + '_test'  + '.csv'
out_var_train = fields['Path'] + '/' + fields['y'] + '_train' +  '.pkl'
out_var_test =  fields['Path'] + '/' + fields['y'] + '_test' + '.pkl'

# check if files exist

if not os.path.isfile(dataset_train) or not os.path.isfile(dataset_test):
    raise OSError("dataset does not exist")

if not os.path.isfile(out_var_train) or not os.path.isfile(out_var_test):
    raise OSError("Output variable does not exist")

# open files

x_train = pd.read_csv(dataset_train)
x_test = pd.read_csv(dataset_test)
y_train = pd.read_pickle(out_var_train)
y_test = pd.read_pickle(out_var_test)

y_train = np.array(y_train)
y_test = np.array(y_test)
setattr(y_train, "shape", (y_train.shape[0], 1))
setattr(y_test, "shape", (y_test.shape[0], 1))

# perform regression on train


line_reg = LinearRegression()
line_reg.fit(x_train, y_train)
predictions = line_reg.predict(x_test)
