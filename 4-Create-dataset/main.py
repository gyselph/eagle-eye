"""Create a dataset from process provenance graphs."""

from graphs_to_windows import GraphsToWindows
from windows_to_dataset import WindowsToDataset

# the settings
GRAPHS_BENIGN = "4-Create-dataset/graphs/benign"
GRAPHS_MALICIOUS = "4-Create-dataset/graphs/malicious"
WINDOWS_BENIGN = "4-Create-dataset/intermediate_data/1_windows_benign.npz"
WINDOWS_MALICIOUS = "4-Create-dataset/intermediate_data/1_windows_malicious.npz"
WINDOW_SIZE = 50
OVERLAP_FACTOR = 2
FEATURE_SCHEMA_CSV = "4-Create-dataset/security_features/security_feature_names.csv"
DATASET = "4-Create-dataset/dataset.npz"

# step 1: turn provenance graphs into fixed-size sequences of behavior events
processor_benign = GraphsToWindows(
        GRAPHS_BENIGN, WINDOWS_BENIGN, WINDOW_SIZE, OVERLAP_FACTOR, FEATURE_SCHEMA_CSV)
processor_benign.call()
processor_malicious = GraphsToWindows(
        GRAPHS_MALICIOUS, WINDOWS_MALICIOUS, WINDOW_SIZE, OVERLAP_FACTOR, FEATURE_SCHEMA_CSV)
processor_malicious.call()

# step 2: turn the behavior sequence into a dataset; do one-hot encoding.
processor = WindowsToDataset(WINDOWS_BENIGN, WINDOWS_MALICIOUS, DATASET, FEATURE_SCHEMA_CSV)
processor.call()
