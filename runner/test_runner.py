import pytest
import os
import sys
import shutil

import numpy as np
import pandas as pd

from greenml.runner.runner import Runner

def generate_test_dataset(training_var, training_lab, test_lab, test_var, renan_case_task):
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
        "task": renan_case_task,
        "y": "b",
        "mod_tokens": ["multi_nb"],
        "folds" : 2,
        "measure" : "pyRAPL"
    }
    return config

# NOTE: accuracy must be an integer between 0 and 100
def create_binary_dataset_with_accuracy(accuracy):
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
    # create the datasets
    return generate_test_dataset(training_var, training_lab, test_lab, test_var, "Binary_Classification")

def clear_test_dataset():
    try:
        shutil.rmtree("./testdset")
    except FileNotFoundError as fe:
        pass

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
    # create the datasets
    return generate_test_dataset(training_var, training_lab, test_lab, test_var, "Multi_Classification")

def test_accuracy_binary():
    clear_test_dataset()
    cf = create_binary_dataset_with_accuracy(35)
    current_runner = Runner(cf)
    results = current_runner.run()
    accmes = results["metrics"]["Naive_Bayes"]["accuracy"]
    # tolerate 1% of difference only
    assert ((accmes-0.35)/0.35) < 0.01
    clear_test_dataset()

def test_accuracy_multimodal():
    clear_test_dataset()
    cf = create_multimodal_dataset_with_accuracy(35)
    current_runner = Runner(cf)
    results = current_runner.run()
    accmes = results["metrics"]["Naive_Bayes"]["accuracy"]
    # tolerate 1% of difference only
    assert ((accmes-0.35)/0.35) < 0.01
    clear_test_dataset()