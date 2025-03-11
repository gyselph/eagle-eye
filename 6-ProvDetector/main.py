# imports
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
from sklearn import metrics

from prov_detector.frequency_db import create_frequency_db
from prov_detector.rarest_paths import create_rarest_paths
from prov_detector.process_paths import convert_paths_to_sentences
from prov_detector.doc2vec import embed_raw_sentences_doc2vec, train_doc2vec_dont_embed
from prov_detector.random_forest import train_rf, test_rf

# Settings
TRAIN_DATA = "data/behavior_events/train"
TEST_DATA = "data/behavior_events/test"
RESULTS = "data/results/"

# Please refer to the ProvDetector and EagleEye paper for hyper parameter settings
NUM_RARE_PATHS_PER_GRAPH_TRAIN = 10
NUM_RARE_PATHS_PER_GRAPH_TEST = 10
SUBPATH_LENGTH_LIMIT = 10
MAX_NUM_SUBPATHS_PER_GRAPH = 10
DOC2VEC_EPOCHS = 10
DOC2VEC_VECTOR_DIM = 10
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 10


def run():
    # Training stage
    frequency_db, entities_db = create_frequency_db(TRAIN_DATA)
    rare_paths_train = create_rarest_paths(frequency_db, entities_db, TRAIN_DATA, NUM_RARE_PATHS_PER_GRAPH_TRAIN)
    _, labels_train, sentences_train = convert_paths_to_sentences(rare_paths_train, SUBPATH_LENGTH_LIMIT, MAX_NUM_SUBPATHS_PER_GRAPH)
    doc2vec_model = train_doc2vec_dont_embed(sentences_train, DOC2VEC_VECTOR_DIM, DOC2VEC_EPOCHS)
    embeddings_train = embed_raw_sentences_doc2vec(doc2vec_model, sentences_train)
    rf_model = train_rf(embeddings_train, labels_train, RF_N_ESTIMATORS, RF_MAX_DEPTH)

    # Test stage
    rare_paths_test = create_rarest_paths(frequency_db, entities_db, TEST_DATA, NUM_RARE_PATHS_PER_GRAPH_TEST)
    graph_ids_test, labels_test, sentences_test = convert_paths_to_sentences(rare_paths_test, SUBPATH_LENGTH_LIMIT,
                                                                             MAX_NUM_SUBPATHS_PER_GRAPH)
    embeddings_test = embed_raw_sentences_doc2vec(doc2vec_model, sentences_test)
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
    run()
