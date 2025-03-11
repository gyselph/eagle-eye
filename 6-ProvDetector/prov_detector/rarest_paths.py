import os
from .common import read_all_events
import math
import bisect
import networkx as nx
import numpy as np
from functools import partial
import multiprocessing as mp
from itertools import islice
from functools import reduce


INDEX_SRC_ID = 0
INDEX_SRC_TYPE = 1
INDEX_DEST_ID = 2
INDEX_DEST_TYPE = 3
INDEX_ACTION = 4
INDEX_PROCESS_NAME = 5
INDEX_TIMESTAMP = 6
INDEX_PID_0 = 7
INDEX_PID_1 = 8

NUM_PROCESSES = 8


def create_rarest_paths(frequency_db, entities_db, event_folder, num_rare_paths_per_graph):
    events_as_dataframe, _, _, graph_labels = read_all_events(event_folder)
    df_graphs = seperate_graphs(events_as_dataframe)
    print("Generating rarest paths database for {} events and {} graphs".format(len(events_as_dataframe), len(df_graphs)))
    with mp.Pool(NUM_PROCESSES) as pool:
        paths_for_each_graph = list(
            pool.imap(
                partial(
                    create_rarest_paths_one_graph,
                    entities_db = entities_db,
                    frequency_db = frequency_db,
                    num_rare_paths_per_graph = num_rare_paths_per_graph,
                    graph_labels = graph_labels
                ),
                df_graphs,
            )
        )
    print("\nFinished computation of rarest paths")
    return paths_for_each_graph


def create_rarest_paths_one_graph(graph_df, entities_db, frequency_db, num_rare_paths_per_graph, graph_labels):
    graphName = graph_df['graphId'].iloc[0]
    print(".", end="")
    event_list = graph_df.values.tolist()
    adjListForward, _ = createAdjListCleanly(event_list, entities_db, frequency_db)
    adjForward = sortTime(adjListForward)
    forAdj, backAdj = makeAdjListDAGFaster(adjForward)
    adjMatrix = shortestPath(forAdj, backAdj)
    kPaths = findKAnomlousPaths(adjMatrix, num_rare_paths_per_graph)
    graph_label = graph_labels[graphName]
    return (graphName, graph_label, kPaths)


def seperate_graphs(df):
    gb = df.groupby('graphId')
    graphs = [gb.get_group(x) for x in gb.groups]
    return graphs


def calculateScore(src, dest, action, allSrcAndDest, srcToDestFrequency):
    src = list(src)
    dest = list(dest)
    retVal = None
    if type(src[0]) != str and np.isnan(src[0]):
        retVal = math.log2(0.5)*-1
    if type(dest[0]) != str and np.isnan(dest[0]):
        retVal = math.log2(0.5)*-1
    if retVal is None:
        inScore = getInScore(src[0], allSrcAndDest)
        outScore = getOutScore(dest[0], allSrcAndDest)
        freqScore = getFreqScore(src[0], dest[0], action, srcToDestFrequency)
        if outScore == 0:
            outScore = 1/len(allSrcAndDest)
        if inScore == 0:
            inScore = 1/len(allSrcAndDest)
        retVal = math.log2(inScore*freqScore*outScore)*-1
    return retVal*-1


def createAdjListCleanly(parsedList, allSrcAndDest, srcToDestFrequency):
    adjListForward = {}
    adjListBackward = {}
    for row in parsedList:
        src, dest = (row[INDEX_SRC_ID],row[INDEX_SRC_TYPE], 0), (row[INDEX_DEST_ID],row[INDEX_DEST_TYPE], 0)
        # source is file or socket
        if row[INDEX_SRC_TYPE] != 'process':
            dest = (row[INDEX_DEST_ID], row[INDEX_PID_0], row[INDEX_DEST_TYPE], 0)
        # destination is file or socket
        elif row[INDEX_DEST_TYPE] != 'process':
            src = (row[INDEX_SRC_ID],row[INDEX_PID_0], row[INDEX_SRC_TYPE], 0)
        # process starts new process
        else:
            src = (row[INDEX_SRC_ID], row[INDEX_PID_0], row[INDEX_SRC_TYPE], 0)
            dest = (row[INDEX_DEST_ID], row[INDEX_PID_1], row[INDEX_DEST_TYPE], 0)
        addToAdjList(src, dest, (row[INDEX_TIMESTAMP], row[INDEX_ACTION], calculateScore((row[INDEX_SRC_ID], row[INDEX_SRC_TYPE]), (row[INDEX_DEST_ID], row[INDEX_DEST_TYPE]), row[INDEX_ACTION], allSrcAndDest, srcToDestFrequency)), adjListForward, adjListBackward)
    return adjListForward, adjListBackward


def makeAdjListDAGFaster(adjListForward):
    forwardEdges = []
    setOfNodes = {}
    dagForAdj = {}
    dagDestAdj = {}
    for src in adjListForward:
        for edge in adjListForward[src]:
            forwardEdges.append((edge[0], src, edge[1], edge[2], edge[3]))
    forwardEdges = sorted(forwardEdges)
    for edge in forwardEdges:
        src = edge[1]
        dest = edge[2]
        edgeAttributes = (edge[0], edge[3], edge[4])
        if dest not in setOfNodes:
            setOfNodes[dest] = 0
        else:
            while setOfNodes.get(dest, 0) == 1:
                dest = list(dest)
                dest[-1] += 1
                dest = tuple(dest)
            setOfNodes[dest] = 0
        if src in setOfNodes:
            while setOfNodes.get(src, 0) == 1:
                src = list(src)
                src[-1] += 1
                src = tuple(src)
            if src in setOfNodes:
                setOfNodes[src] = 1
            else:
                src = list(src)
                src[-1] -= 1
                src = tuple(src)
        else:
            setOfNodes[src] = 1
        dagForAdj.setdefault(src, [])
        dagForAdj[src].append((dest, edgeAttributes))
        dagDestAdj.setdefault(dest, [])
        dagDestAdj[dest].append((src, edgeAttributes))
    return dagForAdj, dagDestAdj


def shortestPath(adjForward, adjBackward):
    adjForward, adjBackward = addSinkSource(adjForward, adjBackward)
    return adjForward


def getInScore(src, allSrcAndDest):
    count = 0
    for index in range(len(allSrcAndDest)):
        nodeSet = allSrcAndDest[index][0]
        if src in nodeSet:
            count += 1
    return count / len(allSrcAndDest)


def getOutScore(dest, allSrcAndDest):
    count = 0
    startIndex = -1
    for index in range(len(allSrcAndDest)):
        nodeSet = allSrcAndDest[index][1]
        if dest in nodeSet:
            if startIndex == -1:
                startIndex = startIndex
            count += 1
    return count / ((len(allSrcAndDest)) - startIndex)


def getFreqScore (src, dest, action, srcToDestFrequency):
    srcRel = (src, action)
    if srcRel not in srcToDestFrequency:
        return 0.001
    if dest not in srcToDestFrequency[srcRel]:
        return 0.001
    return srcToDestFrequency[srcRel][dest] / srcToDestFrequency[srcRel]['total']


def sortTime(adjDict):
    for key in adjDict:
        adjDict[key] = sorted(adjDict[key])
    return adjDict


def addSinkSource(adjForward, adjBackward):
    source = ('source')
    sink = ('sink')
    startSrc = []
    endDest = []
    for src in adjForward:
        if src not in adjBackward:
            startSrc.append(src)
    for dest in adjBackward:
        if dest not in adjForward:
            endDest.append(dest)
    adjForward[source] = []
    adjBackward[sink] = []
    for src in startSrc:
        adjForward[source].append((src, (-1, '(sycal:source)', 0)))
        adjBackward[src] = [((source),(-1, '(sycal:source)', 0))]
    for dest in endDest:
        adjBackward[sink].append((dest, (-1, '(sycal:sink)', 0)))
        adjForward[dest] = [((sink),(-1, '(sycal:sink)', 0))]
    return adjForward, adjBackward


def findKAnomlousPaths(adjMatrix, K):
    """
    Return a list of rare paths, where each path is stored as:
    - A 3-tuple of path, score, and some boolean indicating if this is a DAG graph
    - The path is a list of edges
    - Each edge is a list of source node, interaction type, and target node
    - Each node is a 3-tuple of system entity, entity type, and ?
    - Each interaction type is a 2-tuple of action as String, and PID
    """
    G = nx.DiGraph()
    for src in adjMatrix:
        for row in adjMatrix[src]:
            G.add_edge(src, row[0], weight=row[1][2], action=row[1][1], timestamp=row[1][0])
    isDAG = nx.is_directed_acyclic_graph(G)
    if not isDAG:
        raise Exception("Graph Is Not A DAG")
    Kpaths = []
    adj = G.adj
    for path in k_shortest_paths(G, 'source', 'sink', K, weight='weight'):
        Kpath = []
        regularityScore = 0
        for index in range(len(path)-1):
            try:
                ea = (path[index], path[index+1])
                edgeAttrib = adj[ea[0]][ea[1]]
            except:
                import pdb
                pdb.set_trace()
            regularityScore += edgeAttrib['weight']
            Kpath.append([ea[0],(edgeAttrib['action'], edgeAttrib['timestamp']), ea[1]])
        Kpaths.append([Kpath, regularityScore, isDAG])
    # we computed more paths than necessary, remove some if we have too many
    Kpaths.sort(key = lambda p: p[1])
    Kpaths = Kpaths[:K]
    return Kpaths


def k_shortest_paths(G, source, target, k, weight):
    # this will return the shortest paths in terms of hops, not weights (since we have negative weights, which are not supported by this function)
    short_paths = list(islice(nx.shortest_simple_paths(G, source=source, target=target, weight=None), 2 * k))
    # this will return one short path, and only multiple paths if there is a tie
    short_path_with_weights = list(nx.all_shortest_paths(G, source=source, target=target, weight=weight, method = 'bellman-ford'))
    short_paths.extend(short_path_with_weights)
    # remove duplicates
    short_paths = reduce(lambda re, x: re+[x] if x not in re else re, short_paths, [])
    return short_paths


def addToAdjList(src, dest, edgeAttributes, adjListForward, adjListBackward):
    if src not in adjListForward:
        adjListForward[src] = []
    if dest not in adjListBackward:
        adjListBackward[dest] = []
    srcEdge = list(edgeAttributes)
    destEdge = list(edgeAttributes)
    srcEdge.insert(1, dest)
    destEdge.insert(1, src)
    bisect.insort_left(adjListForward[src], tuple(srcEdge))
    bisect.insort_left(adjListBackward[dest], tuple(destEdge))