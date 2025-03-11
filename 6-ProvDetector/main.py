# imports
import numpy as np
import os
import shutil
import sys
import matplotlib.pyplot as plt
from sklearn import metrics

from prov_detector.frequency_db import create_frequency_db, load_frequency_db
from prov_detector.rarest_paths import create_rarest_paths
from prov_detector.process_paths import convert_paths_to_sentences
from prov_detector.doc2vec import embed_raw_sentences_doc2vec, train_doc2vec_dont_embed
from prov_detector.random_forest import train_rf, test_rf, load_rf_model

# settings and hyper parameters

TRAIN_DATA = "data/behavior_events/train"
FREQUENCY_DB_TRAIN = "data/frequency_db/frequency_db_train.data"
ENTITIES_DB_TRAIN = "data/frequency_db/entities_db_train.data"
RAREST_PATHS_TRAIN_RESULT = "data/rarest_paths/rarest_paths_train.data"
SENTENCE_DB_TRAIN = "data/rarest_paths/sentences_train.npz"
DOC2VEC_MODEL = "data/doc2vec/model.data"
DOC2VEC_EMBEDDINGS_TRAIN = "data/doc2vec/embeddings_train.npz"
RF_MODEL = "data/rf/model.joblib"
TEST_DATA = "data/behavior_events/test"
FREQUENCY_DB_TEST = "data/frequency_db/frequency_db_test.data"
ENTITIES_DB_TEST = "data/frequency_db/entities_db_test.data"
RAREST_PATHS_TEST_RESULT = "data/rarest_paths/rarest_paths_test.data"
SENTENCE_DB_TEST = "data/rarest_paths/sentences_test.npz"
DOC2VEC_EMBEDDINGS_TEST = "data/doc2vec/embeddings_test.npz"
RESULTS = "data/results/"

BENIGN_LABEL = 0
MALICIOUS_LABEL = 1

NUM_RARE_PATHS_PER_GRAPH_TRAIN = 10
NUM_RARE_PATHS_PER_GRAPH_TEST = 10
SUBPATH_LENGTH_LIMIT = 10
MAX_NUM_SUBPATHS_PER_GRAPH = 10
DOC2VEC_EPOCHS = 10
DOC2VEC_VECTOR_DIM = 10
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 10


def train():
    """Perform training of ProvDetector malware detection."""
    frequency_db, entities_db = create_frequency_db(TRAIN_DATA, FREQUENCY_DB_TRAIN, ENTITIES_DB_TRAIN)
    rare_paths_train = create_rarest_paths(frequency_db, entities_db, TRAIN_DATA, RAREST_PATHS_TRAIN_RESULT, NUM_RARE_PATHS_PER_GRAPH_TRAIN)
    _, labels_train, sentences_train = convert_paths_to_sentences(rare_paths_train, SENTENCE_DB_TRAIN, SUBPATH_LENGTH_LIMIT, MAX_NUM_SUBPATHS_PER_GRAPH)
    train_doc2vec_dont_embed(sentences_train, DOC2VEC_MODEL, DOC2VEC_VECTOR_DIM, DOC2VEC_EPOCHS)
    embeddings_train = embed_raw_sentences_doc2vec(DOC2VEC_MODEL, DOC2VEC_EMBEDDINGS_TRAIN, sentences_train)
    train_rf(embeddings_train, labels_train, RF_MODEL, RF_N_ESTIMATORS, RF_MAX_DEPTH)


def test():
    """Run test stage or ProvDetector, by doing malware prediction using trained ML models."""
    frequency_db, entities_db = load_frequency_db(FREQUENCY_DB_TRAIN, ENTITIES_DB_TRAIN)
    rare_paths_test = create_rarest_paths(frequency_db, entities_db, TEST_DATA, RAREST_PATHS_TEST_RESULT, NUM_RARE_PATHS_PER_GRAPH_TEST)
    graph_ids_test, labels_test, sentences_test = convert_paths_to_sentences(rare_paths_test, SENTENCE_DB_TEST, SUBPATH_LENGTH_LIMIT,
                                                                             MAX_NUM_SUBPATHS_PER_GRAPH)
    embeddings_test = embed_raw_sentences_doc2vec(DOC2VEC_MODEL, DOC2VEC_EMBEDDINGS_TEST, sentences_test)
    rf_model = load_rf_model(RF_MODEL)
    predictions_test = test_rf(embeddings_test, rf_model)

    # evaluation
    # ensure we have a fresh results folder
    # Remove the folder if it exists
    folder = RESULTS
    if os.path.exists(folder):
        shutil.rmtree(folder)    
    os.makedirs(folder)
    # compute "predictions" per graph, by taking the average path predictions per graph
    graph_predictions = np.empty(0)
    graph_labels = np.empty(0)
    graph_names = np.empty(0)
    for graph_id in np.unique(graph_ids_test):
        path_predictions = predictions_test[graph_ids_test == graph_id]
        graph_prediction = np.average(np.array(path_predictions))
        graph_predictions = np.append(graph_predictions, graph_prediction)
        graph_label = labels_test[graph_id == graph_ids_test][0]
        graph_labels = np.append(graph_labels, graph_label)
    # Draw ROC curve for binary predictions
    fpr, tpr, _ = metrics.roc_curve(graph_labels, graph_predictions)
    roc_auc = metrics.auc(fpr, tpr)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='ProvDetector')
    display.plot()
    plt.title('ProvDetector graph predictions, supervised, MALICIOUS=1')
    plt.grid(visible=True)
    plt.savefig(folder + "/roc_curve_test.png")
    plt.show()

    binary_predictions = graph_predictions > 0.5
    accuracy = np.mean(binary_predictions == graph_labels)
    print("At graph level, accuracy: {:.3f}\n".format(accuracy))



if __name__ == '__main__':
    arg = sys.argv[1]
    if arg == "train":
        train()
    elif arg == "test":
        test()
    else:
        sys.exit("Need to define command line argument: 'train' or 'test'")
        