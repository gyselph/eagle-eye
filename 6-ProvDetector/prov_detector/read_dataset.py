"""Read in dataset from disk"""

import pandas as pd
from pathlib import Path
from typing import Tuple, Dict


SEPERATOR = ","
BENIGN_MARKER = "benign"
MALICIOUS_MARKER = "malicious"
BENIGN_LABEL = 0
MALICIOUS_LABEL = 1


def read_all_events(event_folder: str) -> Tuple[pd.DataFrame, Dict]:
    """
    Fetch all behavior events from disk, and put them in one big pandas dataframe.

    Each CSV file in `event_folder` represents one graph. The CSV files contain a series of behavior events.

    We expect the following dataframe columns for the results:
    - sourceId: The name of the source system entity
    - sourceType: The type of the source system entity
    - destinationId: The name of the destination system entity
    - destinationType: The type of the destination system entity
    - action: The action type of the behavior event
    - processName: The name of the process (name of executable)
    - timestamp: Timestamp as integer
    - pid0: The process ID of the first involved executable
    - pid1: An optional ID of a second invovlved executable
    - graphId: The ID of the provenance graph, as integer

    :param event_folder: The folder which contains all graphs as .csv files
    :return: A pandas DataFrame with all behavior events in stacked format, plus the label for each graph as dictionary
    """
    csv_files = list(Path(event_folder).rglob("*.csv"))
    csv_files = [str(x) for x in csv_files]
    graph_labels = {}
    list_of_dataframes = []
    print(f"Start reading in all events from {event_folder}.")
    for i in range(len(csv_files)):
        tmp_dataframe = _read_csv_file(csv_files[i], i)
        list_of_dataframes.append(tmp_dataframe)
        label = BENIGN_LABEL if BENIGN_MARKER in csv_files[i] else MALICIOUS_LABEL
        graph_labels[i] = label
    one_big_dataframe = pd.concat(list_of_dataframes)
    print("Finished reading in all events")
    return one_big_dataframe, graph_labels


def _read_csv_file(event_file, graph_id):
    df_events = pd.read_csv(event_file, sep=SEPERATOR, dtype = {'sourceId': str, 'destinationId': str, 'pid0': float, 'pid1': float})
    df_events['graphId'] = graph_id
    return df_events
