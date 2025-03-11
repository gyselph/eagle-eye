import numpy as np
import os
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier


def train_rf(data, labels, num_estimators, max_depth):
    """
    Train a random forest for supervised classification. Inputs: Doc2Vec embeddings. Outputs: binary prediction.
    """
    rf = RandomForestClassifier(n_estimators = num_estimators, max_depth = max_depth, n_jobs = -1)
    print("Start Random Forest training ...")
    rf.fit(data, labels)
    print("Random Forest training has finished")
    return rf


def test_rf(data, model):
    return model.predict(data)


def load_rf_model(model_file):
    return load(model_file)


def store_rf_model(model, model_file):
    dump(model, model_file) 