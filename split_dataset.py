"""
split datasets script
"""

import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataset_location(dataset_location, variable, recoding=False,
                  recoding_index=None):
    """Splits dataset_locations into train test
    output variables are apppended to the end of the
    data frame
    
    :parameter pd.DataFrame dataset_location: Dataset_Location to perform the split on
    :parameter str variable: Output variable
    :parameter bool recoding: Boolean indicating whether to recode or not
    :parameter set recoding_index: index of variables to recode
    :returns: tuple of train, test dataset_locations 
    :rtype: tuple
    :raises ValueError: If dataset_location and variable are not strings

    """
    if not isinstance(dataset_location, str):
        raise ValueError("dataset_location must be a string")
    if not isinstance(variable, str):
        raise ValueError("variable must be a string")
    X = pd.read_csv(dataset_location)
    y = X[f"{variable}"]
    X = df.drop(variable, axis=1)

    if recoding:
        ## Insert code to recode nominal
        ## variables here
        pass

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2)
    print(y_train.shape)
    X_train = pd.concat([X_train, y_train], axis=1)
    X_test = pd.concat([X_train, y_train], axis=1)

    return X_train, X_test

