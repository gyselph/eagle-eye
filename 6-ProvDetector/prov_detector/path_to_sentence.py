"""Convert rare paths in graphs to english sentences."""

import math
import numpy as np
from typing import List, Tuple

SOURCE_INDEX = 0
ACTION_INDEX = 1
DESTINATION_INDEX = 2
NODE_ID_INDEX = 0
NODE_TYPE_INDEX = -2
INDEX_LABEL = 1
INDEX_PATHS = 2
WORD_SEPERATOR = " ||| "


def paths_to_sentences(rare_paths: List[Tuple], chunk_size_limit: int, max_num_chunks: int):
    """
    Convert rare paths to english sentences.
    Processing steps include: chunking of overly long paths, translation of rare paths from complex
      data format to english sentences, some data cleanup.

    :param rare_paths: Rare paths in complex data format
    :param chunk_size_limit: Max length of a path, split up overly longer paths
    :param max_num_chunks: The maximal number of chunks per graph
    :return: A tuple consisting of 1) graph IDs, 2) labels, 3) sentences

    The rarest paths have the following format:
    The rare paths are a list of paths per graph. Each path is a tuple consisting of
      1) the graph ID
      2) the graph label
      3) a list of rare paths
    The rare paths themselves are a list of nodes, each of which consists of:
      1) A source system entity (id, optional: pid, type, and ?)
      2) An action (action type, timestamp)
      3) A target system entity (id, optional: pid, type, and ?)
    """
    graph_ids, chunk_labels, chunks = _chunk_paths(rare_paths, chunk_size_limit, max_num_chunks)
    sentences = [_chunk_to_sentence(s) for s in chunks]
    return np.array(graph_ids), np.array(chunk_labels), np.array(sentences)


def _chunk_paths(rare_paths, chunk_size_limit, max_num_chunks):
    graph_ids = []
    chunk_labels = []
    chunks = []
    for graph_id in range(len(rare_paths)):
        graph_label = rare_paths[graph_id][INDEX_LABEL]
        graph_paths = rare_paths[graph_id][INDEX_PATHS]
        chunks_this_graph = []
        for path_id in range(len(graph_paths)):
            path = graph_paths[path_id][0]
            # drop special source and destination nodes
            path = path[1:-1]
            # do chunking when path is long
            num_parts = math.ceil(len(path) / chunk_size_limit)
            for i in range(num_parts):
                chunks_this_graph.append(path[i*chunk_size_limit:(i+1)*chunk_size_limit])
        # ensure we don't exceed the max number of chunks per graph
        chunks_this_graph = chunks_this_graph[:max_num_chunks]
        graphs_ids_this_graph = [graph_id] * len(chunks_this_graph)
        labels_this_graph = [graph_label] * len(chunks_this_graph)
        graph_ids.extend(graphs_ids_this_graph)
        chunk_labels.extend(labels_this_graph)
        chunks.extend(chunks_this_graph)
    return graph_ids, chunk_labels, chunks


def _cleanup_action(action, destination_type):
    if destination_type == "process" and action == 'read':
        action = action + "_by"
    return action


def _chunk_to_sentence(path):
    first_source_id = path[0][SOURCE_INDEX][NODE_ID_INDEX]
    first_source_type = path[0][SOURCE_INDEX][NODE_TYPE_INDEX]
    sentence = f"{first_source_type}:{first_source_id}"
    for event_index in range(len(path)):
        action = path[event_index][ACTION_INDEX][0]
        destination = path[event_index][DESTINATION_INDEX]
        destination_id = destination[0]
        destination_type = destination[NODE_TYPE_INDEX]
        action = _cleanup_action(action, destination_type)
        addition = WORD_SEPERATOR + action + WORD_SEPERATOR + destination_type + ":" + destination_id
        sentence += addition
    return sentence
