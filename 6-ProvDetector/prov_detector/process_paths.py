import math
import os
import numpy as np


SOURCE_INDEX = 0
ACTION_INDEX = 1
DESTINATION_INDEX = 2
NODE_ID_INDEX = 0
NODE_TYPE_INDEX = -2
ACTIONS = ["read", "write", "start"]
ENTITY_TYPES = ["process", "file", "socket"]
INDEX_GRAPH_NAME = 0
INDEX_GRAPH_LABEL = 1
INDEX_PATHS = 2
WORD_SEPERATOR = " ||| " # must NOT be present in dataset!

def convert_paths_to_sentences(rare_paths, subpath_length_limit, max_num_subpaths_per_graph):
    """
    Do all preprocessing on paths:
    - split huge paths into manageable sub-paths
    - translate paths from complex data format to english sentences
    - do some cleanup on data, like normalizing user names and removing local IP addresses

    Parameters:
        - rare_paths: Rare paths in proprietary format
        - path_length_limit: Max length of a path, split up longer paths
        - max_num_subpaths: The maximal number of sub-paths we use, per graph
    
    Return:
        - graph_names: The graph name per sentence
        - labels: The label per sentence (0 or 1)
        - sentences: The rare paths, translated to an English sentences
    """
    graph_names, sub_path_labels, sub_paths = chunk_paths(rare_paths, subpath_length_limit, max_num_subpaths_per_graph)
    sentences = [pathToSentence(s) for s in sub_paths]
    return np.array(graph_names), np.array(sub_path_labels), np.array(sentences)


def chunk_paths(rare_paths, subpath_length_limit, max_num_subpaths):
    graph_names = []
    sub_path_labels = []
    sub_paths = []
    for graph_id in range(len(rare_paths)):
        graph_name = rare_paths[graph_id][INDEX_GRAPH_NAME]
        graph_label = rare_paths[graph_id][INDEX_GRAPH_LABEL]
        graph_paths = rare_paths[graph_id][INDEX_PATHS]
        graph_names_this_graph = []
        labels_this_graph = []
        sub_paths_this_graph = []
        for path_id in range(len(graph_paths)):
            path = graph_paths[path_id][0]
            # drop special source and destination nodes
            path = path[1:-1]
            num_parts = math.ceil(len(path) / subpath_length_limit)
            for i in range(num_parts):
                start = i*subpath_length_limit
                end = min(len(path), (i+1)*subpath_length_limit)
                graph_names_this_graph.append(graph_name)
                labels_this_graph.append(graph_label)
                sub_paths_this_graph.append(path[start:end])
        if max_num_subpaths > 0:
            # TODO: are the rarest paths really sorted by anomaly score? (this is an assumption here)
            limit = min(len(sub_paths_this_graph), max_num_subpaths)
            graph_names_this_graph = graph_names_this_graph[:limit]
            labels_this_graph = labels_this_graph[:limit]
            sub_paths_this_graph = sub_paths_this_graph[:limit]
        graph_names.extend(graph_names_this_graph)
        sub_path_labels.extend(labels_this_graph)
        sub_paths.extend(sub_paths_this_graph)
    return graph_names, sub_path_labels, sub_paths


def cleanup_system_entity(id, type):
    if type not in ENTITY_TYPES:
        print("Unknown type: {}".format(type))
    if str(id) == 'nan' and math.isnan(id):
        # TODO figure out what goes wrong here
        print("Bad ID: {}".format(id))
        id = "*"
    return (id, type)


def cleanup_action(action, destination_type):
    # make sure we have valid action type
    if action not in ACTIONS:
        print("Unknown type: {}".format(action))
    if destination_type == "process" and action == 'read':
        action = action + "_by"
    return action


def pathToSentence(path):
    sentence = ""
    first_source_id = path[0][SOURCE_INDEX][NODE_ID_INDEX]
    first_source_type = path[0][SOURCE_INDEX][NODE_TYPE_INDEX]
    first_source_id, first_source_type = cleanup_system_entity(first_source_id, first_source_type)
    sentence += "{}:{}".format(first_source_type, first_source_id)
    for event_index in range(len(path)):
        action = path[event_index][ACTION_INDEX]
        destination = path[event_index][DESTINATION_INDEX]
        action = action[0]
        destination_id = destination[0]
        destination_type = destination[-2]
        destination_id, destination_type = cleanup_system_entity(destination_id, destination_type)
        action = cleanup_action(action, destination_type)
        addition = WORD_SEPERATOR + action + WORD_SEPERATOR + destination_type + ":" + destination_id
        sentence += addition
    return sentence

