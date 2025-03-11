TRAIN_DATA = "experiments_darpa/events/train/"
FREQUENCY_DB_TRAIN = "experiments_darpa/frequency_db/frequency_db_train.data"
ENTITIES_DB_TRAIN = "experiments_darpa/frequency_db/entities_db_train.data"
RAREST_PATHS_TRAIN_RESULT = "experiments_darpa/rarest_paths/rarest_paths_train.data"
SENTENCE_DB_TRAIN = "experiments_darpa/rarest_paths/sentences_train.npz"
DOC2VEC_MODEL = "experiments_darpa/doc2vec/model.data"
DOC2VEC_EMBEDDINGS_TRAIN = "experiments_darpa/doc2vec/embeddings_train.npz"
RF_MODEL = "experiments_darpa/rf/model.joblib"
TEST_DATA = "experiments_darpa/events/test"
FREQUENCY_DB_TEST = "experiments_darpa/frequency_db/frequency_db_test.data"
ENTITIES_DB_TEST = "experiments_darpa/frequency_db/entities_db_test.data"
RAREST_PATHS_TEST_RESULT = "experiments_darpa/rarest_paths/rarest_paths_test.data"
SENTENCE_DB_TEST = "experiments_darpa/rarest_paths/sentences_test.npz"
DOC2VEC_EMBEDDINGS_TEST = "experiments_darpa/doc2vec/embeddings_test.npz"
RESULTS = "experiments_darpa/results/2023-11-22/"

BENIGN_LABEL = 0
MALICIOUS_LABEL = 1

# hyper parameters
NUM_RARE_PATHS_PER_GRAPH_TRAIN = 100
NUM_RARE_PATHS_PER_GRAPH_TEST = 20
SUBPATH_LENGTH_LIMIT = 10
MAX_NUM_SUBPATHS_PER_GRAPH = 20
DOC2VEC_EPOCHS = 100
DOC2VEC_VECTOR_DIM = 20
RF_N_ESTIMATORS = 2000
RF_MAX_DEPTH = 20

OVERSAMPLE_MALICIOUS_GRAPHS_TRAIN = 8
UNDERSAMPLE_MALICIOUS_SENTENCES_TRAIN = 1
OVERSAMPLE_MALICIOUS_GRAPHS_TEST = 8
UNDERSAMPLE_MALICIOUS_SENTENCES_TEST = 0.7


from frequency_db import create_frequency_db, load_frequency_db
from rarest_paths import create_rarest_paths
from process_paths import convert_paths_to_sentences
from oversampling import oversample_malicious_paths, create_graph_ids_to_name_map
from doc2vec import embed_raw_sentences_doc2vec, train_doc2vec_dont_embed
from random_forest import train_rf, test_rf, load_rf_model
import numpy as np
import string
import random
import os
import sys
import matplotlib.pyplot as plt
from sklearn import metrics
import pandas as pd


def train():
    # run training data through pipeline
    # ----------------------------------
    frequency_db, entities_db = create_frequency_db(TRAIN_DATA, FREQUENCY_DB_TRAIN, ENTITIES_DB_TRAIN, False)
    rare_paths_train = create_rarest_paths(frequency_db, entities_db, TRAIN_DATA, RAREST_PATHS_TRAIN_RESULT, NUM_RARE_PATHS_PER_GRAPH_TRAIN, False)
    graph_ids_to_names = create_graph_ids_to_name_map(TRAIN_DATA)
    graph_ids_train, labels_train, sentences_train = convert_paths_to_sentences(rare_paths_train, SENTENCE_DB_TRAIN, SUBPATH_LENGTH_LIMIT, -1, False)
    graph_names_train = [graph_ids_to_names[id] for id in graph_ids_train]
    graph_ids_train_oversampled, graph_names_train_oversampled, labels_train_oversampled, sentences_train_oversampled = oversample_malicious_paths(
        graph_ids_train, graph_names_train, labels_train, sentences_train, OVERSAMPLE_MALICIOUS_GRAPHS_TRAIN, UNDERSAMPLE_MALICIOUS_SENTENCES_TRAIN)
    train_doc2vec_dont_embed(sentences_train, DOC2VEC_MODEL, DOC2VEC_VECTOR_DIM, DOC2VEC_EPOCHS, False)
    embeddings_train_oversampled = embed_raw_sentences_doc2vec(DOC2VEC_MODEL, DOC2VEC_EMBEDDINGS_TRAIN, sentences_train_oversampled, False)
    train_rf(embeddings_train_oversampled, labels_train_oversampled, RF_MODEL, RF_N_ESTIMATORS, RF_MAX_DEPTH, False)


def test():
    # run test data through pipeline
    # ------------------------------
    frequency_db, entities_db = load_frequency_db(FREQUENCY_DB_TRAIN, ENTITIES_DB_TRAIN)
    rare_paths_test = create_rarest_paths(frequency_db, entities_db, TEST_DATA, RAREST_PATHS_TEST_RESULT, NUM_RARE_PATHS_PER_GRAPH_TEST, False)
    graph_ids_to_names = create_graph_ids_to_name_map(TEST_DATA)
    graph_ids_test, labels_test, sentences_test = convert_paths_to_sentences(rare_paths_test, SENTENCE_DB_TEST, SUBPATH_LENGTH_LIMIT,
                                                                             MAX_NUM_SUBPATHS_PER_GRAPH, False)
    graph_names_test = [graph_ids_to_names[id] for id in graph_ids_test]
    graph_ids_test_oversampled, graph_names_test_oversampled, labels_test_oversampled, sentences_test_oversampled = oversample_malicious_paths(
        graph_ids_test, graph_names_test, labels_test, sentences_test, OVERSAMPLE_MALICIOUS_GRAPHS_TEST, UNDERSAMPLE_MALICIOUS_SENTENCES_TEST)
    embeddings_test_oversampled = embed_raw_sentences_doc2vec(DOC2VEC_MODEL, DOC2VEC_EMBEDDINGS_TEST, sentences_test_oversampled, False)
    rf_model = load_rf_model(RF_MODEL)
    predictions_test_oversampled = test_rf(embeddings_test_oversampled, rf_model)
    
    predictions_test = predictions_test_oversampled
    graph_ids_test = np.array(graph_ids_test_oversampled)
    graph_names_test = np.array(graph_names_test_oversampled)
    labels_test = np.array(labels_test_oversampled)

    # evaluation
    # ----------
    folder = RESULTS + ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
    os.mkdir(folder)
    print("Storing test results in {}".format(folder))
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
        graph_name = graph_names_test[graph_id == graph_ids_test][0]
        graph_names = np.append(graph_names, graph_name)
    # Draw ROC curve for binary predictions
    fpr, tpr, _ = metrics.roc_curve(graph_labels, graph_predictions)
    roc_auc = metrics.auc(fpr, tpr)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='ProvDetector')
    display.plot()
    plt.title('ProvDetector graph predictions, supervised, MALICIOUS=1')
    plt.xscale('log')
    plt.grid(visible=True)
    plt.savefig(folder + "/roc_curve_test.png")
    plt.show()
    # Compute TPR at FPR key thresholds
    index_1_in_10_fpr = np.argmax(fpr > 0.1) - 1
    index_1_in_100_fpr = np.argmax(fpr > 0.01) - 1
    index_1_in_1000_fpr = np.argmax(fpr > 0.001) - 1
    binary_predictions = graph_predictions > 0.5
    accuracy = np.mean(binary_predictions == graph_labels)
    with open(folder + "/results_test.txt", 'w') as f:
        f.write("At graph level, false positive rate: {:.3f}, true positive rate: {:.3f}\n".format(fpr[index_1_in_10_fpr], tpr[index_1_in_10_fpr]))
        f.write("At graph level, false positive rate: {:.3f}, true positive rate: {:.3f}\n".format(fpr[index_1_in_100_fpr], tpr[index_1_in_100_fpr]))
        f.write("At graph level, false positive rate: {:.3f}, true positive rate: {:.3f}\n".format(fpr[index_1_in_1000_fpr], tpr[index_1_in_1000_fpr]))
        f.write("At graph level, AUC: {:.3f}\n".format(roc_auc))
        f.write("At graph level, accuracy: {:.3f}\n".format(accuracy))
    # Store hyper parameters
    with open(folder + "/config.txt", 'w') as f:
        f.write("""NUM_RARE_PATHS_PER_GRAPH_TRAIN: {}, NUM_RARE_PATHS_PER_GRAPH_TEST: {}, SUBPATH_LENGTH_LIMIT: {}, MAX_NUM_SUBPATHS_PER_GRAPH: {},
                DOC2VEC_EPOCHS: {}, DOC2VEC_VECTOR_DIM: {}, RF_N_ESTIMATORS: {}, RF_MAX_DEPTH: {}, TRAIN_DATA: {}\n""".format(
            NUM_RARE_PATHS_PER_GRAPH_TRAIN, NUM_RARE_PATHS_PER_GRAPH_TEST, SUBPATH_LENGTH_LIMIT, MAX_NUM_SUBPATHS_PER_GRAPH, DOC2VEC_EPOCHS,
            DOC2VEC_VECTOR_DIM, RF_N_ESTIMATORS, RF_MAX_DEPTH, TRAIN_DATA
        ))
    # Store ROC curve - raw data
    np.savez_compressed(folder + "/roc.npz", tpr = tpr, fpr = fpr)
    # Store prediction per graph
    errors = np.abs(graph_predictions - graph_labels)
    pd.DataFrame({"graph_name": graph_names, "errors": errors}).to_csv(folder + "/prediction_errors.csv", index=False)


if __name__ == '__main__':
    arg = sys.argv[1]
    if arg == "train":
        train()
    elif arg == "test":
        test()
    else:
        sys.exit("Need to define command line argument: 'train' or 'test'")
        