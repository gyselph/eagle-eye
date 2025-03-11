# Re-implementation of ProvDetector

This is a re-implementation of [ProvDetector](https://kangkookjee.io/wp-content/uploads/2021/06/provdetector-ndss2020.pdf)

Repos used as reference include:
- Mimicry paper repo: https://bitbucket.org/sts-lab/mimicry-provenance-generator/src/master/
- Some random repo from Tsinghua students: https://github.com/nodiff-229/GAT_provdetector/blob/master/main.py

There are two versions of ProvDetector: the original unsupervised version, and our adaption to a supervised version.

## Pipeline overview

Graph classification with ProvDetector works as follows:
1. Turn graphs from our data format into the ProvDetector data format (CSV)
2. On train data only:
 * Create a frequency database for the train data, where we store how frequent one system entity interact with another one
 * Compute lots of (rare) paths from the train graphs, and turn those into readable sentences, like `process:winword.exe write file:t1.txt read_by process:outlook.exe write socket:101.22.22.1:8080`.
 * Train a document embedding model on sentences (Doc2Vec)
3. Learn final predictions on training data:
 - For supervised learning: Train a random forest classifier, which classifies path embeddings into benign and malicious.
 - For unsupervised learning: Fit a LOF model in "novelty detection" mode.
4. On the test data:
 * Compute rare paths from the graphs, using the frequency database from the train data
 * Apply the already trained document embedding model
 * Compute final prediction: 
   * For supervised learning: Pass path embeddings to the random forest to predict the graph class.
   * For unsupervised learning: Use an outlier model to check if the test embeddings are far away from all train embeddings (local outlier factor)

## ProvDetector data format (CSV)

|Explanation|sourceId|sourceType|destinationId|destinationType|action|processName|timestamp|pid0|pid1|
|---|---|---|---|---|---|---|---|---|---|
|Column meaning|0: Full source name|1: Source type|2: Full destination name|3: Destination type|4: Action type|5: Name of (first) executable|6: Timestamp|7: (first) PID|8: optional, second PID|
|File read|/data/file.txt|file|/bin/proc1.exe|process|read|proc1.exe|1000|1||
|File write|/bin/proc1.exe|process|/data/file.txt|file|write|proc1.exe|1001|1||
|Socket send|/bin/proc1.exe|process|192.168.0.1|socket|send|proc1.exe|1002|1||
|Socket receive|192.168.0.1|socket|/bin/proc1.exe|process|recv|proc1.exe|1003|1||
|Start new process|/bin/proc1.exe|process|/bin/proc2.exe|process|execve|proc1.exe|1004|1|2|
|Start new process|/bin/proc2.exe|process|/bin/proc2.exe|process|clone|proc2.exe|1005|2|20|

## Run

- Convert a bunch of graphs from our data format to the ProvDetector data format: Run `data_conversion.main.py`. Make sure to set the source and target folders properly. This script will also reshuffle the data split, and make sure all users appear in the training data.
- Run the ProvDetector train & test pipeline. Make sure to set all hyper parameters properly.
  - For supervised learning: Run `main_supervised.py` with `train` and `test` arguments.
  - For unsupervised learning: Run `main_unsupervised.py` with `train` and `test` arguments.

Configurations in `main.py`:
- `TRAIN_DATA`: The CSV data used for creating the frequency database. Moreover, this data is also used to create training paths.
- `FREQUENCY_DB_TRAIN` and `ENTITIES_DB_TRAIN`: The file path where the frequency database gets stored.
- `RAREST_PATHS_TRAIN_RESULT`: The train paths get stored in pickle format.
- `SENTENCE_DB_TRAIN`: The train sentences get stored as compressed numpy array.
- `DOC2VEC_MODEL` and `DOC2VEC_EMBEDDINGS_TRAIN`: The document embedding model and embedding vectors get stored as pickled object and as compressed numpy array.
- `RF_MODEL`: The random forest classifier gets trained on train data and stored in this file location.
- `LOF_MODEL`: The outlier model gets trained on train data and stored in this file location.
- `TEST_DATA`: The CSV test data.
- `RAREST_PATHS_TEST_RESULT`: Same as `RAREST_PATHS_TRAIN_RESULT`, but for test data set.
- `SENTENCE_DB_TEST`: Same as `SENTENCE_DB_TRAIN`, but for test data set.
- `DOC2VEC_EMBEDDINGS_TEST`: Same as `DOC2VEC_EMBEDDINGS_TRAIN`, but for test data set.
- `RESULTS`: Dump all results into this folder.

The contents of train and test CSV folders should look as follows:
- `TRAIN_DATA`: this folder can have any file structure
- `TEST_DATA`: this folder should have a `benign` and a `malicious` subfolder, with CSV files in both folders.

Hyperparameters in `main_*.py`:
- `NUM_RARE_PATHS_PER_GRAPH_TRAIN`: Number of rare paths that get constructed per graph, for the training dataset.
- `NUM_RARE_PATHS_PER_GRAPH_TEST`: Same as `NUM_RARE_PATHS_PER_GRAPH_TRAIN`, but for test data set.
- `SUBPATH_LENGTH_LIMIT`: Rarely, paths are very long. If this happens, we break them up into sub-paths of this maximal length.
- `MAX_NUM_SUBPATHS_PER_GRAPH`: The maximal number of sub-paths per graph.
- `DOC2VEC_EPOCHS`: The number of training epochs for the Doc2Vec algorithm.
- `DOC2VEC_VECTOR_DIM`: The dimension of the document embedding vector.
- `RF_N_ESTIMATORS`: The number of decision trees in the random forest.
- `RF_MAX_DEPTH`: The maximal depth per decision tree.
- `LOF_N_NEIGHBORS`: The number of neighbors used for the outlier model. The locality of a given point is compared to the locality of the `LOF_N_NEIGHBORS` neighbors.
- `LOF_TRAIN_CONTAMINATION`: The contamination assumed for training of the outlier model. Higher values mean more sensitivity at test time, and thus more outliers.

## Possible improvements

- Avoid bias from CAPE sandboxes: Make sure there are really no references to `amf_susp.exe` or the `AMF` user in the dataset. Specifically, each file and socket event contains its parent process, which can be the original `amf_susp.exe` executable.
- Re-visit how we generate rare paths. Do originally proposed function `nx.all_shortest_paths(..., 'bellman-ford')` doesn't return as many paths as promised by documentation.