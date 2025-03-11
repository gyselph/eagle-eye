import pandas as pd
import pickle
from pathlib import Path


SEPERATOR = ","
BENIGN = "benign"
MALICIOUS = "malicious"
BENIGN_LABEL = 0
MALICIOUS_LABEL = 1


GRAPHS_NOT_WORKING = [
    "malicious_02525fb8a9d2d1726dffa8edcb947f3f418727c40111a480d11a71d23e907db7.be_logs.zip_amf_susp.exe.3019362016736.txt.csv"
]


def read_all_events(event_folder):
    csv_files = list(Path(event_folder).rglob("*.csv"))
    csv_files = [str(path).replace("\\", "/") for path in csv_files]
    graph_labels = {}
    list_of_dataframes, list_of_events, list_of_graph_ids = [], [], []
    print("Start reading in all events from {}".format(event_folder))
    for i in range(len(csv_files)):
        if csv_files[i][csv_files[i].rfind("/")+1:] in GRAPHS_NOT_WORKING:
            print("Skipping graph that doesn't work for some weird reason: {}".format(csv_files[i]))
            continue
        tmp_dataframe, tmp_events, tmp_graph_ids = readPandasFile(csv_files[i], i)
        list_of_dataframes.append(tmp_dataframe)
        list_of_events.extend(tmp_events)
        list_of_graph_ids.extend(tmp_graph_ids)
        label = BENIGN_LABEL if BENIGN in csv_files[i] else MALICIOUS_LABEL
        graph_labels[i] = label
    one_big_dataframe = pd.concat(list_of_dataframes)
    print("Finished reading in all events")
    return one_big_dataframe, list_of_events, list_of_graph_ids, graph_labels


def readPandasFile(event_file, graph_id):
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


def write_object_to_file(jsonObjectList, fileName):
    with open(fileName, 'wb') as filehandle:
        pickle.dump(jsonObjectList, filehandle)


def read_object_from_file(fileName):
    with open(fileName, 'rb') as filehandle:
        # read the data as binary data stream
        jsonObjectList = pickle.load(filehandle)
    return jsonObjectList