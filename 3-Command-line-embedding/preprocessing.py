import random
import json
import os
import time
from pathlib import Path
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


DATA_DIR = 'data'


def load_graphs(path: str, max_len: int = None):
    """
    Load the graphs from the given path
    :param path: relative path to the directory containing the graphs
    :param max_len: maximum number of graphs to load. If none, all graphs will be loaded
    """
    base_path = os.path.join(DATA_DIR, path)
    files = list(Path(base_path).rglob("*.[jJ][sS][oO][nN]"))
    max_len = max_len or len(files)
    graphs = []
    users = []
    user_counts = {}
    print(f"Loading {max_len} graphs from {base_path}")
    for path in tqdm(files[:max_len]):
        with open(path) as f:
            js_graph = json.load(f)
        graph = nx.readwrite.tree_graph(js_graph)
        graphs.append(graph)
    return graphs, users, user_counts


def get_root(graph: nx.DiGraph):
    """
    Get the root node of the graph
    :param graph: networkX graph
    """
    for n, d in graph.in_degree():
        if d == 0:
            return n


def extract_cmds(graphs, cmd_key='auxilary_cmdline_for_sentence_transformer') -> list[list[str]]:
    """
    Extract commandlines from the new common dataset
    :param graphs: list of networkX provenance graphs
    :param cmd_key: the key to retrieve cmd property from graphs
    """
    print('Extracting command lines from graphs')
    cmdlines = []
    beg = time.time()
    tree_levels = []
    for graph in graphs:
        root = get_root(graph)
        cmds = []
        levels = []
        data = dict(graph.nodes.data())

        for n, n_d in data.items():
            if n_d.get(cmd_key):
                cmds.append(n_d.get(cmd_key))
                levels.append(nx.shortest_path_length(graph, source=root, target=n))
        if len(cmds) > 0:
            cmdlines.append(cmds)
            tree_levels.append(max(levels))
    print("CMD extraction time: ", time.time() - beg)
    print(f'Number of graphs with commands: {len(cmdlines)} out of {len(graphs)}')
    print('Avg nodes :', np.mean([len(g) for g in graphs]))
    print(f'Average tree depth:  {np.mean(tree_levels)}')
    return cmdlines


def cmd2vec(graphs: list) -> list[np.ndarray]:
    """
    Convert the cmdlines to vectors using sentence transformers
    :param graphs: list of networkX graphs
    :return: Vector representation of graphs
    """

    cmdlines = extract_cmds(graphs=graphs)
    lens = [len(line) for line in cmdlines]
    print('CMDs extracted')
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = []
    print('Computing embeddings')
    for i, line in enumerate(tqdm(cmdlines)):
        emb = model.encode(line)
        embeddings.extend(emb)

    print('embeddings computed')
    print(f'Average tree size:  {np.mean(lens)}')

    return embeddings


def prepare_data():
    """
    Prepare the data for training. Transform command line strings from precess provenance graphs into
    high dimensional embedding vectors using the sentence transformer.

    Provenance graphs should be stored in the data/graphs directory in the following structure:
        data/graphs/malicious/train - contains the training set of malicious graphs
        data/graphs/malicious/val - contains the validation set of malicious graphs
        data/graphs/malicious/test - contains the test set of malicious graphs
        data/graphs/benign/train - contains the training set of benign graphs
        data/graphs/benign/val - contains the validation set of benign graphs
        data/graphs/benign/test - contains the test set of benign graphs
    The function will save the extracted features as csv files in the data/preprocessed directory.
    return: benign training, validation, test sets and malicious training, validation, test sets
    """
    begin = time.time()
    np.random.seed(100)
    random.seed(100)
    all_splits = []
    for dtype in ['malicious', 'benign']:
        splits = []
        os.makedirs(f'{DATA_DIR}/preprocessed/{dtype}', exist_ok=True)
        for split in ['train', 'val', 'test']:
            graphs, _, _ = load_graphs(f'graphs/{dtype}/{split}')
            data = cmd2vec(graphs)
            features = pd.DataFrame(data)
            features.columns = features.columns.astype(str)
            path = f'{DATA_DIR}/preprocessed/{dtype}/{split}_features.csv'
            features.to_csv(path)
            print(f'{dtype} {split} features saved to {path}')
            splits.append(features)
        all_splits.append(splits)

    malicious_features, benign_features = all_splits

    ben_train, ben_val, ben_test = benign_features
    mal_train, mal_val, mal_test = malicious_features

    print("train / val / test split: {}({}) / {} / {}".format(
        len(ben_train) + len(mal_train),
        len(ben_train),
        len(ben_val) + len(mal_val),
        len(ben_test) + len(mal_test))
    )

    benign_samples = len(benign_features)
    malicious_samples = len(malicious_features)
    print("malicious samples per dataset: {}".format(malicious_samples))
    print("benign samples per dataset: {}".format(benign_samples))
    print(f"TIME ELAPSED: {(time.time() - begin) / 60} minutes")
    return ben_train, ben_val, ben_test, mal_train, mal_val, mal_test