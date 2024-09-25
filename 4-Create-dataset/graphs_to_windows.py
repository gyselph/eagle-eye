import os
import json
import random
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import numpy as np
from networkx.readwrite import json_graph
import pandas as pd

@dataclass
class GraphsToWindows:
    """
    Turn graphs to a window dataset.

    Represent each graph by a series of behavior events. For each behavior event, only keep the security features.
    Also, slice the behavior sequences into fixed-size windows.

    Attributes:
        graphs_folder:      Folder with all graphs.
        target_dataset:     File path under which the resulting dataset gets stored.
        window_size:        The number of events for one window.
        overlap_factor:     The amount of overlap between windows. A value of 2 means every event gets into two windows, on average.
        feature_schema_csv: A CSV file with a list of security features plus their default values.
    """

    graphs_folder: str
    target_dataset: str
    window_size: int
    overlap_factor: float
    feature_schema_csv: str


    def _default_val_correct_type(self, default_val: str) -> str | int | float | bool:
        """
        Turn default values for security features from String format into the correct data type.
        """
        if default_val == "NOT_PRESENT":
            return "NOT_PRESENT"
        if default_val == "0":
            return int(0)
        if default_val == "0.0":
            return float(0)
        return False


    def _fetch_feature_schema(self, feature_schema_csv: str) -> List[Tuple[str, Any]]:
        """
        Fetch the list of security feature names and their default values.

        :param feature_schema_csv: A CSV file with columns "security_feature_name" and "default_value".
        :return A list of security feature names plus their respective default value.
        """
        all_features = pd.read_csv(feature_schema_csv)
        all_features = all_features.to_numpy().T
        feature_names = all_features[0]
        default_values = all_features[1]
        default_values = list(map(self._default_val_correct_type, default_values))
        return list(zip(feature_names, default_values))

    def _event_to_feature_vector(self, event: Dict) -> List:
        """
        Turn one behavior event into a feature vector.

        :param dict event: A dictionary of raw features and security features.
        :return list: A vector of security features with a well known size.
        """
        feature_vector = []
        for feature_name, default_value in self._feature_schema:
            if feature_name in event:
                pick = event[feature_name]
            else:
                pick = default_value
            feature_vector.append(pick)
        return feature_vector


    def _graph_to_feature_vectors(self, graph_file: str) -> np.ndarray:
        """
        Turn a provenance graph into a list of feature vectors. Each feature
        vector represents one node in the graph.

        :param graph_file: The file path to the graph.
        :return: A 2d NumPy array of shape (num_nodes, num_features).
        """
        # read graph
        with open(file = graph_file, mode = "r", encoding = "UTF-8") as f:
            json_data = json.load(f)
        graph = json_graph.tree_graph(json_data)
        # sort events by timestamp
        events = [attrs for (_, attrs) in graph.nodes.data() if "timestamp" in attrs]
        events.sort(key = lambda x: x["timestamp"])
        # convert all events to feature vectors
        feature_vectors = [self._event_to_feature_vector(event) for event in events]
        # turn list of feature vectors to numpy matrix
        feature_vectors = np.array(feature_vectors, dtype = 'O')
        return feature_vectors


    def _sequence_to_windows(self, feature_vectors: np.ndarray) -> np.ndarray:
        """
        Turn a long sequence of events into fix-sized windows.

        :param feature_vectors: A 2d array which contains a list of events; shape: (num_events, num_features).
        :return: A 3d array which contains a list of windows, where each window has the same number of events; shape: (num_windows, window_size, num_features).
        """
        # crop windows from the sequence
        sequence_length = feature_vectors.shape[0]
        # special case: not enough data for one full window
        if sequence_length < self.window_size:
            return np.empty((0,self.window_size,feature_vectors.shape[2]))
        # crop windows from the long sequence
        num_of_windows = int(self.overlap_factor * sequence_length / self.window_size)
        windows = []
        for _ in range(num_of_windows):
            start = random.randint(0, sequence_length - self.window_size)
            end = start + self.window_size
            window = feature_vectors[start:end]
            windows.append(window)
        return np.array(windows)


    def call(self) -> np.array:
        """
        Convert all graphs into one window dataset.

        :return: A 3d array of shape (num_windows, window_size, num_features).
        """
        # initialization
        self._feature_schema = self._fetch_feature_schema(self.feature_schema_csv)
        # find all graphs
        graph_files = os.listdir(self.graphs_folder)
        graph_files = [Path(self.graphs_folder) / Path(g) for g in graph_files]
        # turn graph to windows
        print(f"Start processing {len(graph_files)} graphs from {self.graphs_folder} ...")
        sequence_for_each_graph = [self._graph_to_feature_vectors(g) for g in graph_files]
        windows_for_each_graph = [self._sequence_to_windows(s) for s in sequence_for_each_graph]
        windows_filtered = list(filter(lambda windows: len(windows) > 0, windows_for_each_graph))
        # create one big array with all windows
        windows = np.vstack(windows_filtered)
        # persist window dataset
        print(f"Conversion to windows has finished, storing result to {self.target_dataset}.")
        Path(self.target_dataset).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.target_dataset, windows=windows)
        return windows
