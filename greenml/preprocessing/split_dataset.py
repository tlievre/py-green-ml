"""
split datasets script
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder


def split_dataset_location(dataset_location, variable, recoding=False,
                           recoding_index=[]):
    """Splits dataset_locations into train test
    output variables are apppended to the end of the
    data frame
    
    :parameter pd.DataFrame dataset_location: Dataset_Location to perform
    the split on
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
    y = X[variable]
    X = X.drop(variable, axis=1)
    # extract indexes of columns containing
    # strings
    label_indexes = [i if X.dtypes[i].name in {'object', 'string'} else -1
                     for i in range(X.dtypes.shape[0])]
    if set(label_indexes) != {-1}:
        label_indexes = set(label_indexes)
        label_indexes.remove(-1)
        label_indexes = list(label_indexes)
        label_encoder = LabelEncoder()
        for idx in label_indexes:
            # recode each string into
            # integers
            encoded_idx = label_encoder.fit_transform(X.iloc[:, idx])
            X[X.columns[idx]] = encoded_idx
            # recoding_index contains index of
            # categorical variables to be recoded
            # with one hot encoding
    if recoding:
        # ectracting numerical and categorical
        # variables
        recoding_index = recoding_index + label_indexes
        x_num = X.drop(X.columns[recoding_index], axis=1)
        x_nom = X.iloc[:, recoding_index]
        ohe = OneHotEncoder(handle_unknown="ignore")
        # recoding categorical variables using one hot
        # encoding
        ohe = ohe.fit_transform(x_nom)
        ohe_variables = pd.DataFrame.sparse.from_spmatrix(ohe)
        X = pd.concat([ohe_variables, x_num], axis=1)
    x_train, x_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2)
    x_train = pd.concat([x_train, y_train], axis=1)
    x_test = pd.concat([x_train, y_train], axis=1)

    return x_train, x_test
