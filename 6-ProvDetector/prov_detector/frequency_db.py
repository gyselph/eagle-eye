"""Create a frequency map for system entities and their interactions."""

from typing import Tuple, Dict, List
from .read_dataset import read_all_events


SRC_INDEX_IN_CSV = 0
DST_INDEX_IN_CSV = 2
ACTION_INDEX_IN_CSV = 4
GRAPH_INDEX_IN_CSV = 9


def create_frequency_db(event_folder: str) -> Tuple[Dict,List]:
    """
    Main entry point to compute frequency DB.

    :param event_folder: The folder where all provenance graphs are located.
    :return: A connection frequency plus a list of system entities per graph

    The connection frequency is a dictionary of system relationships. The dictionary contains one entry per system entity.
    Each such entry contains a second dictionary of connected system entities with their frequency as value.
    Example connection frequency entry: ('notepad.exe', 'read') -> {'total': 3, 'file1.txt': 2, 'file2.txt': 1} 
    
    The system entity list is computed for every graph individually, and contains a set of all source and target system entities.
    Example system entity entry for a given graph: [{'notepad.exe'}, {'file1.txt', 'file2.txt'}]
    """
    events_as_dataframe, _ = read_all_events(event_folder)
    events_as_list = events_as_dataframe.values.tolist()
    print(f"Generating frequency database for {len(events_as_list)} events ...")
    list_of_graph_ids = sorted(events_as_dataframe['graphId'].unique())
    connection_frequency = {}
    system_entities = [[set(), set()] for _ in range(len(list_of_graph_ids))]
    for event in events_as_list:
        _update_system_entities(event, list_of_graph_ids, system_entities)
        _update_connection_frequency(event, connection_frequency)
    return connection_frequency, system_entities


def _update_system_entities(event, list_of_graph_ids, system_entities):
    graph_id = list_of_graph_ids.index(event[GRAPH_INDEX_IN_CSV])
    src, dest = event[SRC_INDEX_IN_CSV], event[DST_INDEX_IN_CSV]
    if src not in system_entities[graph_id][0]:
        system_entities[graph_id][0].add(src)
    if dest not in system_entities[graph_id][1]:
        system_entities[graph_id][1].add(dest)


def _update_connection_frequency(event, connection_frequency):
    src, dest = event[SRC_INDEX_IN_CSV], event[DST_INDEX_IN_CSV]
    src_and_action = (src, event[ACTION_INDEX_IN_CSV])
    if src_and_action not in connection_frequency:
        connection_frequency[src_and_action] = {}
        connection_frequency[src_and_action]['total'] = 0
    if dest not in connection_frequency[src_and_action]:
        connection_frequency[src_and_action][dest] = 0
    connection_frequency[src_and_action][dest] += 1
    connection_frequency[src_and_action]['total'] += 1
