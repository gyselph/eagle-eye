"""Re-implementation of ProvDetector.
Extract rare paths from provenance graphs, embed them with Doc2Vec, and train a random forest for malware detection.
Original paper: https://kangkookjee.io/wp-content/uploads/2021/06/provdetector-ndss2020.pdf
References for re-implementation:
- Mimicry paper repo: https://bitbucket.org/sts-lab/mimicry-provenance-generator/src/master/provDetector/
- Some random GitHub repo: https://github.com/nodiff-229/GAT_provdetector/blob/master/main.py
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

from prov_detector.frequency_db import create_frequency_db
from prov_detector.rarest_paths import create_rarest_paths
from prov_detector.path_to_sentence import paths_to_sentences
from prov_detector.doc2vec import embed_doc2vec, train_doc2vec
from prov_detector.random_forest import train_rf, test_rf

# Put all training data into this folder. Ensure you have a "benign" and a "malicious" folder within.
TRAIN_DATA = "6-ProvDetector/behavior_events/train"
# Put all test data into this folder. Ensure you have a "benign" and a "malicious" folder within.
TEST_DATA = "6-ProvDetector/behavior_events/test"

# Please refer to the ProvDetector and EagleEye paper for ideal hyper parameter settings
NUM_RARE_PATHS_PER_GRAPH_TRAIN = 10
NUM_RARE_PATHS_PER_GRAPH_TEST = 10
SUBPATH_LENGTH_LIMIT = 10
MAX_NUM_SUBPATHS_PER_GRAPH = 10
DOC2VEC_EPOCHS = 10
DOC2VEC_VECTOR_DIM = 10
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 10


def run():
    """Main entry point for ProvDetector training and testing."""

    # Training stage
    frequency_db, entities_db = create_frequency_db(TRAIN_DATA)
    rare_paths_train = create_rarest_paths(frequency_db, entities_db, TRAIN_DATA, NUM_RARE_PATHS_PER_GRAPH_TRAIN)
    _, labels_train, sentences_train = paths_to_sentences(rare_paths_train, SUBPATH_LENGTH_LIMIT, MAX_NUM_SUBPATHS_PER_GRAPH)
    doc2vec_model = train_doc2vec(sentences_train, DOC2VEC_VECTOR_DIM, DOC2VEC_EPOCHS)
    embeddings_train = embed_doc2vec(doc2vec_model, sentences_train)
    rf_model = train_rf(embeddings_train, labels_train, RF_N_ESTIMATORS, RF_MAX_DEPTH)

    # Test stage
    rare_paths_test = create_rarest_paths(frequency_db, entities_db, TEST_DATA, NUM_RARE_PATHS_PER_GRAPH_TEST)
    graph_ids_test, labels_test, sentences_test = paths_to_sentences(rare_paths_test, SUBPATH_LENGTH_LIMIT,
                                                                             MAX_NUM_SUBPATHS_PER_GRAPH)
    embeddings_test = embed_doc2vec(doc2vec_model, sentences_test)
    predictions_test = test_rf(embeddings_test, rf_model)

    # Compute metrics
    # Compute "predictions" per graph, by taking the average path predictions per graph
    graph_predictions = np.empty(0)
    graph_labels = np.empty(0)
    for graph_id in np.unique(graph_ids_test):
        path_predictions = predictions_test[graph_ids_test == graph_id]
        graph_prediction = np.average(np.array(path_predictions))
        graph_predictions = np.append(graph_predictions, graph_prediction)
        graph_label = labels_test[graph_id == graph_ids_test][0]
        graph_labels = np.append(graph_labels, graph_label)
    # Compute accuracy
    binary_predictions = graph_predictions > 0.5
    accuracy = np.mean(binary_predictions == graph_labels)
    print("Graph classification accuracy: {:.3f}\n".format(accuracy))
    # Draw ROC curve for binary predictions
    fpr, tpr, _ = metrics.roc_curve(graph_labels, graph_predictions)
    roc_auc = metrics.auc(fpr, tpr)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='ProvDetector')
    display.plot()
    plt.title('ProvDetector malware prediction, MALICIOUS=1')
    plt.grid(visible=True)
    plt.show()


if __name__ == '__main__':
    run()

