import pandas as pd
from pathlib import Path


SEPERATOR = ","
BENIGN = "benign"
MALICIOUS = "malicious"
BENIGN_LABEL = 0
MALICIOUS_LABEL = 1


def read_all_events(event_folder):
    csv_files = list(Path(event_folder).rglob("*.csv"))
    csv_files = [str(path).replace("\\", "/") for path in csv_files]
    graph_labels = {}
    list_of_dataframes, list_of_events, list_of_graph_ids = [], [], []
    print("Start reading in all events from {}".format(event_folder))
    for i in range(len(csv_files)):
        tmp_dataframe, tmp_events, tmp_graph_ids = read_csv_file(csv_files[i], i)
        list_of_dataframes.append(tmp_dataframe)
        list_of_events.extend(tmp_events)
        list_of_graph_ids.extend(tmp_graph_ids)
        label = BENIGN_LABEL if BENIGN in csv_files[i] else MALICIOUS_LABEL
        graph_labels[i] = label
    one_big_dataframe = pd.concat(list_of_dataframes)
    print("Finished reading in all events")
    return one_big_dataframe, list_of_events, list_of_graph_ids, graph_labels


def read_csv_file(event_file, graph_id):
    df_events = pd.read_csv(event_file, sep=SEPERATOR, dtype = {'sourceId': str, 'destinationId': str, 'pid0': float, 'pid1': float})
    df_events['graphId'] = graph_id
    df_events['sourceId'] = df_events['sourceId'].str.strip('" \n\t')
    df_events['destinationId'] = df_events['destinationId'].str.strip('" \n\t')
    df_events['action'] = df_events['action'].str.strip('" \n\t')
    df_events['sourceType'] = df_events['sourceType'].str.strip('" \n\t')
    df_events['destinationType'] = df_events['destinationType'].str.strip('" \n\t')
    parsedList = df_events.values.tolist()
    uniqueGraphNames = sorted(list(df_events.graphId.unique()))
    return df_events, parsedList, uniqueGraphNames
