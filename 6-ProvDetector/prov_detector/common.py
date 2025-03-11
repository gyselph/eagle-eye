import pandas as pd
from pathlib import Path


SEPERATOR = ","
BENIGN_MARKER = "benign"
MALICIOUS_MARKER = "malicious"
BENIGN_LABEL = 0
MALICIOUS_LABEL = 1


def read_all_events(event_folder):
    csv_files = list(Path(event_folder).rglob("*.csv"))
    csv_files = [str(x) for x in csv_files]
    graph_labels = {}
    list_of_dataframes = []
    print("Start reading in all events from {}".format(event_folder))
    for i in range(len(csv_files)):
        tmp_dataframe = read_csv_file(csv_files[i], i)
        list_of_dataframes.append(tmp_dataframe)
        label = BENIGN_LABEL if BENIGN_MARKER in csv_files[i] else MALICIOUS_LABEL
        graph_labels[i] = label
    one_big_dataframe = pd.concat(list_of_dataframes)
    list_of_events = one_big_dataframe.values.tolist()
    list_of_graph_ids = list(range(len(csv_files)))
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
    return df_events
