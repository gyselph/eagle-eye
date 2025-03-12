"""Train a random forest to detect malware based on rare path embeddings."""

from sklearn.ensemble import RandomForestClassifier
import numpy as np

def train_rf(training_data: np.ndarray, labels: np.ndarray, num_estimators: int, max_depth: int) -> RandomForestClassifier:
    """
    Train a random forest for supervised classification.
    Inputs: Doc2Vec embeddings. Outputs: binary prediction.

    :param training_data: Training data of shape (number of rare paths, embedding dimension)
    :param labels: Binary labels of shape (number of rare paths,)
    :param num_estimators: Hyper parameter: number of trees for the forest
    :param max_depth: Hyper parameter: the maximal depth per tree
    """
    rf = RandomForestClassifier(n_estimators = num_estimators, max_depth = max_depth, n_jobs = -1)
    print("Start Random Forest training ...")
    rf.fit(training_data, labels)
    print("Random Forest training has finished")
    return rf


def test_rf(test_data: np.ndarray, model: RandomForestClassifier) -> np.ndarray:
    """Run inference on the trained random forest.
    
    :param test_data: Test data of shape (number of rare paths, embedding dimension)
    :param model: The trained random forest
    :return: The prediction data of shape (number of rare paths,)
    """
    return model.predict(test_data)
