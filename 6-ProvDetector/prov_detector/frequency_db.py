import numpy as np
import os
from .read_dataset import read_all_events


GRAPH_INDEX_IN_CSV = 9


def create_frequency_db(event_folder):
    # read in events from all event CSV files
    events_as_dataframe, _ = read_all_events(event_folder)
    events_as_list = events_as_dataframe.values.tolist()
    list_of_graph_ids = sorted(events_as_dataframe['graphId'].unique())
    print("Generating frequency database for {} events ...".format(len(events_as_list)))
    frequency_db, entities_db = createFreqDict(events_as_list, list_of_graph_ids, GRAPH_INDEX_IN_CSV)
    return frequency_db, entities_db


def createFreqDict(parsedList, listOfGraphs, graphIndex):
    """
    Main entry point to compute frequency DB.

    Return:
    - srcToDestFrequency: A dictionary of sources, where for each source we get a list of all destinations, and frequent they are
    - allSrcAndDest: For each graph, create a source and a destination set
    """
    allSrcAndDest = []
    for _ in range(len(listOfGraphs)):
        allSrcAndDest.append([set(), set()])
    srcToDestFrequency = {}
    for row in parsedList:
        addOneSrcAndDest(row, listOfGraphs, allSrcAndDest, graphIndex)
        src, dest = row[0], row[2]
        if type(src) != str and np.isnan(src):
            src = 'None'
        if type(dest) != str and np.isnan(dest):
            dest = 'None'
        srcRel = (src, row[4])
        if srcRel not in srcToDestFrequency:
            srcToDestFrequency[srcRel] = {}
            srcToDestFrequency[srcRel]['total'] = 0
        if dest not in srcToDestFrequency[srcRel]:
            srcToDestFrequency[srcRel][dest] = 0
        srcToDestFrequency[srcRel][dest] += 1
        srcToDestFrequency[srcRel]['total'] += 1
    return srcToDestFrequency, allSrcAndDest


def addOneSrcAndDest(row, listOfGraphs, allSrcAndDest, graphIndex):
    index = listOfGraphs.index(row[graphIndex])
    src, dest = row[0], row[2]
    if type(src) != str and np.isnan(src):
        src = 'None'
    if type(dest) != str and np.isnan(dest):
        dest = 'None'
    if src not in allSrcAndDest[index][0]:
        allSrcAndDest[index][0].add(src)
    if dest not in allSrcAndDest[index][1]:
        allSrcAndDest[index][1].add(dest)

