"""Create rarest paths from graphs and frequency map."""

from .read_dataset import read_all_events
import math
import bisect
import networkx as nx
import numpy as np
from itertools import islice
from functools import reduce
from typing import Dict, List

INDEX_SRC_ID = 0
INDEX_SRC_TYPE = 1
INDEX_DEST_ID = 2
INDEX_DEST_TYPE = 3
INDEX_ACTION = 4
INDEX_TIMESTAMP = 6
INDEX_PID_0 = 7
INDEX_PID_1 = 8


def create_rarest_paths(connection_frequency: Dict, system_entities: List, event_folder: str, num_rare_paths_per_graph: int) -> List:
    """
    Create rarest paths from graphs and frequency map.

    :param connection_frequency: The frequency at which a given source interacts with a given destination.
    :param system_entities: A list of all system entities, per graph.
    :return: The rarest paths in complex data format.

    The connection frequency is a dictionary of system relationships. The dictionary contains one entry per system entity.
    Each such entry contains a second dictionary of connected system entities with their frequency as value.
    Example connection frequency entry: ('notepad.exe', 'read') -> {'total': 3, 'file1.txt': 2, 'file2.txt': 1} 

    The system entity list contains one entry per graph. Each entry contains a set of all source and target system entities.
    Example system entity entry for a given graph: [{'notepad.exe'}, {'file1.txt', 'file2.txt'}]

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
    # read all graphs from disk
    events_as_dataframe, graph_labels = read_all_events(event_folder)
    # seperate all events by graph id
    gb = events_as_dataframe.groupby('graphId')
    df_graphs = [gb.get_group(x) for x in gb.groups]
    # compute rarest paths for each graph
    print(f"Generating rarest paths database for {len(events_as_dataframe)} events and {len(df_graphs)} graphs.")
    paths = [create_rarest_paths_one_graph(g, connection_frequency, system_entities, num_rare_paths_per_graph, graph_labels) for g in df_graphs]
    print("\nFinished computation of rarest paths")
    return paths


def create_rarest_paths_one_graph(graph_df, connection_frequency, system_entities, num_rare_paths_per_graph, graph_labels):
    """Create all rarest paths for one given graph."""
    print(".", end="")
    graph_id = graph_df['graphId'].iloc[0]
    event_list = graph_df.values.tolist()
    adjacency_list_fwd = create_adjacency_list_from_events(event_list, connection_frequency, system_entities)
    adjacency_list_fwd = sort_time(adjacency_list_fwd)
    adj_list_fwd_dag, adj_list_bwd_dag = create_adjacency_list_dag(adjacency_list_fwd)
    add_sink_and_source(adj_list_fwd_dag, adj_list_bwd_dag)
    rarest_paths = find_rarest_paths(adj_list_fwd_dag, num_rare_paths_per_graph)
    graph_label = graph_labels[graph_id]
    return (graph_id, graph_label, rarest_paths)


def create_adjacency_list_from_events(event_list, connection_frequency, system_entities):
    """Create an adjaceency list from all events in one graph.

    :return: The forward adjacency list.
    
    The adjacency list is a dictionary of nodes, where the values are a list of edges:
    - nodes are in format (event id, event type, 0)
    - edges are in format (timestamp, (event id, optional: pid, event type, 0), action, rareness score)
    """
    adjacency_list_fwd = {}
    for event in event_list:
        # Create the source and destination node. Every node has an ID and a type. On top, processes have
        # a process ID.
        # Source is not process, target is process:
        if event[INDEX_SRC_TYPE] != 'process':
            src = (event[INDEX_SRC_ID], event[INDEX_SRC_TYPE], 0)
            dest = (event[INDEX_DEST_ID], event[INDEX_PID_0], event[INDEX_DEST_TYPE], 0)
        # Target is not process, source is process:
        elif event[INDEX_DEST_TYPE] != 'process':
            src = (event[INDEX_SRC_ID], event[INDEX_PID_0], event[INDEX_SRC_TYPE], 0)
            dest = (event[INDEX_DEST_ID], event[INDEX_DEST_TYPE], 0)
        # Both source and target are processes
        else:
            src = (event[INDEX_SRC_ID], event[INDEX_PID_0], event[INDEX_SRC_TYPE], 0)
            dest = (event[INDEX_DEST_ID], event[INDEX_PID_1], event[INDEX_DEST_TYPE], 0)
        rareness_score = calculate_rareness_score((event[INDEX_SRC_ID], event[INDEX_SRC_TYPE]), (event[INDEX_DEST_ID], event[INDEX_DEST_TYPE]), event[INDEX_ACTION], system_entities, connection_frequency)
        edge_attributes = (event[INDEX_TIMESTAMP], event[INDEX_ACTION], rareness_score)
        add_event_to_adjacency_list(src, dest, edge_attributes, adjacency_list_fwd)
    return adjacency_list_fwd


def create_adjacency_list_dag(adjacency_list_fwd):
    """Create a two adjacency lists: one with forward edges, one with backward edges."""
    forward_edges = []
    set_of_nodes = {}
    adj_list_fwd_dag = {}
    adj_list_bwd_dag = {}
    for src in adjacency_list_fwd:
        for edge in adjacency_list_fwd[src]:
            forward_edges.append((edge[0], src, edge[1], edge[2], edge[3]))
    forward_edges = sorted(forward_edges)
    for edge in forward_edges:
        src = edge[1]
        dest = edge[2]
        edge_attributes = (edge[0], edge[3], edge[4])
        if dest not in set_of_nodes:
            set_of_nodes[dest] = 0
        else:
            while set_of_nodes.get(dest, 0) == 1:
                dest = list(dest)
                dest[-1] += 1
                dest = tuple(dest)
            set_of_nodes[dest] = 0
        if src in set_of_nodes:
            while set_of_nodes.get(src, 0) == 1:
                src = list(src)
                src[-1] += 1
                src = tuple(src)
            if src in set_of_nodes:
                set_of_nodes[src] = 1
            else:
                src = list(src)
                src[-1] -= 1
                src = tuple(src)
        else:
            set_of_nodes[src] = 1
        adj_list_fwd_dag.setdefault(src, [])
        adj_list_fwd_dag[src].append((dest, edge_attributes))
        adj_list_bwd_dag.setdefault(dest, [])
        adj_list_bwd_dag[dest].append((src, edge_attributes))
    return adj_list_fwd_dag, adj_list_bwd_dag


def add_sink_and_source(adj_list_fwd_dag, adj_list_bwd_dag):
    """Add special source and sink nodes to the DAG.
    
    Source nodes are connected to nodes without incoming edges.
    Sink nodes are connected to nodes without outgoing edges.
    """
    source = ('source')
    sink = ('sink')
    start_src = []
    end_dest = []
    for src in adj_list_fwd_dag:
        if src not in adj_list_bwd_dag:
            start_src.append(src)
    for dest in adj_list_bwd_dag:
        if dest not in adj_list_fwd_dag:
            end_dest.append(dest)
    adj_list_fwd_dag[source] = []
    adj_list_bwd_dag[sink] = []
    for src in start_src:
        adj_list_fwd_dag[source].append((src, (-1, '(sycal:source)', 0)))
        adj_list_bwd_dag[src] = [((source),(-1, '(sycal:source)', 0))]
    for dest in end_dest:
        adj_list_bwd_dag[sink].append((dest, (-1, '(sycal:sink)', 0)))
        adj_list_fwd_dag[dest] = [((sink),(-1, '(sycal:sink)', 0))]
    return adj_list_fwd_dag, adj_list_bwd_dag


def find_rarest_paths(adj_list_fwd_dag, num_rare_paths_per_graph):
    """
    Return a list of rare paths, where each path is stored as:
    - A 3-tuple of path, score, and some boolean indicating if this is a DAG graph
    - The path is a list of edges
    - Each edge is a list of source node, interaction type, and target node
    - Each node is a 3-tuple of system entity, entity type, and ?
    - Each interaction type is a 2-tuple of action as String, and PID
    """
    # Convert adjacency list to networkx di-graph
    G = nx.DiGraph()
    for src in adj_list_fwd_dag:
        for row in adj_list_fwd_dag[src]:
            G.add_edge(src, row[0], weight=row[1][2], action=row[1][1], timestamp=row[1][0])
    is_DAG = nx.is_directed_acyclic_graph(G)
    rarest_paths = []
    adj = G.adj
    k_rarest_paths = k_shortest_paths(G, 'source', 'sink', num_rare_paths_per_graph, weight='weight')
    for path in k_rarest_paths:
        rare_path = []
        rareness_score = 0
        for index in range(len(path)-1):
            ea = (path[index], path[index+1])
            edge_attributes = adj[ea[0]][ea[1]]
            rareness_score += edge_attributes['weight']
            rare_path.append([ea[0],(edge_attributes['action'], edge_attributes['timestamp']), ea[1]])
        rarest_paths.append([rare_path, rareness_score, is_DAG])
    # we computed more paths than necessary, remove some if we have too many
    rarest_paths.sort(key = lambda p: p[1])
    rarest_paths = rarest_paths[:num_rare_paths_per_graph]
    return rarest_paths


def calculate_rareness_score(src, dest, action, system_entities, connection_frequency):
    """Calculate the rareness score for a given source, destination, and action."""
    src = list(src)
    dest = list(dest)
    source_score = get_source_score(src[0], system_entities)
    destination_score = get_destination_score(dest[0], system_entities)
    frequency_score = get_frequency_score(src[0], dest[0], action, connection_frequency)
    if destination_score == 0:
        destination_score = 1/len(system_entities)
    if source_score == 0:
        source_score = 1/len(system_entities)
    rareness_score = math.log2(source_score*frequency_score*destination_score)*-1
    return rareness_score*-1


def get_source_score(src, system_entities):
    """Get a rareness score based on how frequent this source appears in interactions."""
    count = 0
    for index in range(len(system_entities)):
        node_set = system_entities[index][0]
        if src in node_set:
            count += 1
    return count / len(system_entities)


def get_destination_score(dest, system_entities):
    """Get a rareness score based on how frequent this destination appears in interactions."""
    count = 0
    start_index = -1
    for index in range(len(system_entities)):
        node_set = system_entities[index][1]
        if dest in node_set:
            if start_index == -1:
                start_index = start_index
            count += 1
    return count / ((len(system_entities)) - start_index)


def get_frequency_score(src, dest, action, connection_frequency):
    """Compute a rareness score according to how frequent the destination is interacting with the source."""
    src_relationship = (src, action)
    if src_relationship not in connection_frequency:
        return 0.001
    if dest not in connection_frequency[src_relationship]:
        return 0.001
    return connection_frequency[src_relationship][dest] / connection_frequency[src_relationship]['total']


def sort_time(adjacency_list_fwd):
    """For each node, sort all edges by timestamp."""
    for key in adjacency_list_fwd:
        adjacency_list_fwd[key] = sorted(adjacency_list_fwd[key])
    return adjacency_list_fwd


def add_event_to_adjacency_list(src, dest, edge_attributes, adjacency_list_fwd):
    """
    Add a new edge to the forward adjacency list.

    src / dest: (event id, optional: pid, event type, 0)
    edge_attributes: (timestamp, action, rareness_score)
    adjacency_list_*: a dictionary of nodes, where the values are a list of edges
      nodes are in format (event id, event type, 0)
      edges are in format (timestamp, (event id, optional: pid, event type, 0), action, rareness score)
    """
    if src not in adjacency_list_fwd:
        adjacency_list_fwd[src] = []
    src_edge = list(edge_attributes)
    src_edge.insert(1, dest)
    bisect.insort_left(adjacency_list_fwd[src], tuple(src_edge))


def k_shortest_paths(G, source, target, k, weight):
    """Compute shortest / most rare path using networkx function all_shortest_paths()."""
    # this will return the shortest paths in terms of hops, not weights (since we have negative weights, which are not supported by this function)
    short_paths = list(islice(nx.shortest_simple_paths(G, source=source, target=target, weight=None), 2 * k))
    # this will return one short path, and only multiple paths if there is a tie
    short_path_with_weights = list(nx.all_shortest_paths(G, source=source, target=target, weight=weight, method = 'bellman-ford'))
    short_paths.extend(short_path_with_weights)
    # remove duplicates
    short_paths = reduce(lambda re, x: re+[x] if x not in re else re, short_paths, [])
    return short_paths
