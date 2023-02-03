import pytest
import os
import sys

import numpy as np
import pandas as pd

from runner.runner import Runner

# NOTE: accuracy must be an integer between 0 and 100
def create_simple_dataset_with_accuracy(accuracy):
    # abort if accuracy is invalid
    if not isinstance(accuracy, int) or accuracy < 0 or accuracy > 100:
        print("invalid accuracy")
        raise ValueError("Invalid accuracry provided")
    # create training set
    training_var = np.repeat(np.concatenate([np.ones(40, dtype=np.int64),np.zeros(24, dtype=np.int64)], dtype=np.int64), 100)
    training_lab = np.repeat(np.concatenate([np.ones(40, dtype=np.int64),np.zeros(24, dtype=np.int64)], dtype=np.int64), 100)
    # create test set
    test_lab = np.ones(10000, dtype=np.int64)
    test_var = np.repeat(np.concatenate([np.ones(accuracy, dtype=np.int64),np.zeros(100-accuracy, dtype=np.int64)], dtype=np.int64), 100)
    # create pandas entries
    test_ds = pd.DataFrame({
        "a": test_var,
        "b": test_lab
    })
    train_ds = pd.DataFrame({
        "a": training_var,
        "b": training_lab
    })
    # create folder
    os.mkdir("./testdset")
    # write dataset and config to disk
    test_ds.to_csv("testdset/testdset_test.csv")
    train_ds.to_csv("testdset/testdset_train.csv")
    # create config
    config = {
        "name": "testdset",
        "path": "./testdset/",
        "task": "Binary_Classification",
        "y": "b",
        "mod_tokens": ["d_tree"],
        "folds" : 2,
        "measure" : "pyRAPL"
    }
    return config

def clear_test_dataset():
    os.rmdir("./testdset")

def create_multimodal_dataset_with_accuracy(accuracy):
    # abort if accuracy is invalid
    if not isinstance(accuracy, int) or accuracy < 0 or accuracy > 100:
        print("invalid accuracy")
        raise ValueError("Invalid accuracry provided")
    # create training set
    training_var = np.repeat(np.concatenate([np.ones(40, dtype=np.int64), np.zeros(24, dtype=np.int64), np.repeat(3, 24)], dtype=np.int64), 100)
    training_lab = np.repeat(np.concatenate([np.ones(40, dtype=np.int64), np.zeros(24, dtype=np.int64), np.repeat(3, 24)], dtype=np.int64), 100)
    # create test set
    test_lab = np.ones(10000, dtype=np.int64)
    test_var = np.repeat(np.concatenate([np.ones(accuracy, dtype=np.int64),np.zeros(100-accuracy, dtype=np.int64)], dtype=np.int64), 100)
    # create pandas entries
    test_ds = pd.DataFrame({
        "a": test_var,
        "b": test_lab
    })
    train_ds = pd.DataFrame({
        "a": training_var,
        "b": training_lab
    })
    # create folder
    os.mkdir("./testdset")
    # write dataset and config to disk
    test_ds.to_csv("testdset/testdset_test.csv")
    train_ds.to_csv("testdset/testdset_train.csv")
    # create config
    config = {
        "name": "testdset",
        "path": "./testdset/",
        "task": "Binary_Classification",
        "y": "b",
        "mod_tokens": ["d_tree"],
        "folds" : 2,
        "measure" : "pyRAPL"
    }
    return config

def test_dataset():
    cf = create_simple_dataset_with_accuracy(35)
    current_runner = Runner(config_file)
    results = current_runner.run()
    accmes = results["metrics"]["Decision_Tree"]["accuracy"]
    # tolerate 1% of difference only
    assert ((accmes-0.35)/0.35) < 0.01