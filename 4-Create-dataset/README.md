# Create a dataset from process provenance graphs

This folder contains all code necessary to transform process provenance graphs into an ML training dataset. The input of this data pipeline is a set of process provenance, enriched with security features, all represented in JSON format. The output is a numpy dataset, split into a *training* and *validation* part, where each sample is a behavior sequence of fixed size.

## Run the code

- Navigate to the repository root directory
- Use Python 3.12
- Create a pip virtual environment:
```
python -m venv .venv
source ./.venv/bin/activate
```
- Install required Python libraries:
```
pip install -r ./requirements.txt
```
- Run the main script:
```
python ./4-Create-dataset/main.py
```

This code uses a small set of 4 sample graphs for dataset creation. Note that you'll need ~1k graphs to create a reasonable dataset.

## Input and output

**The input**:

As input, we use process provenance graphs in JSON format. The provenance graphs need to be enriched with security features, prior to running this data pipeline step. Sample graphs can be found under [4-Create-dataset/graphs/](./graphs/). The graphs are ready to be used by the [networkx](https://networkx.org/documentation/stable/index.html) library. You can import them using the `networkx.readwrite.json_graph()` function.

Here's how you can load a given graph and analyze it with `networkx`:

```
import json
from networkx.readwrite import json_graph
import pprint
graph_file = './4-Create-dataset/graphs/benign/POWERPNT.EXE.8615704416628.txt.json'
with open(file = graph_file, mode = "r", encoding = "UTF-8") as f:
    json_data = json.load(f)
graph = json_graph.tree_graph(json_data)
print(f"Number of graph nodes: {len(graph)}.")
second_node_attributes = graph.nodes['8615704416628.191773']
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(second_node_attributes)
```

The provenance graphs are in tree format, with the root process at the tree head. Each node represents a behavior event. The node analyzed above has the following event features:

```
{
    "event_uuid_v4": "53efae41-78c4-47a5-8eb7-7ecd344a55c0",
    "open_options": "32",
    "open_access_flags": "1179785",
    "event_type": "si_create_file",
    "object_name": "\\Device\\HarddiskVolume4\\WINDOWS\\Prefetch\\POWERPNT.EXE-396E33B0.pf",
    "timestamp": "1665131422361421",
    "create_file_disposition": "1",
    "thread_uid": "-28000448774016",
    "encoding_file_path": "WINDOWS",
    "encoding_open_options": "32",
    "encoding_open_access_flags": "1179785",
    "encoding_create_file_disposition": "1",
    "id": "8615704416628.191773"
}
```

This event represents a file access, and contains a bunch of *raw* features, such as `event_uuid_v4`, `open_options`, etc. Additionally, there event contains *security* features, which we'll use for creating the dataset. Security features in this case include `encoding_file_path`, `encoding_open_access_flags`, etc.

An additional necessary input is the security feature schema. A sample schema can be found under [4-Create-dataset/security_features/security_feature_names.csv](./security_features/security_feature_names.csv). This schema contains a list of security features, plus their default values. For dataset creation, we'll include all `security` features, and drop all `raw` features.

**The output**:

The result of this data pipeline step is a dataset, which is ready for ML training. You can use it for the next step [5-EagleEye-training](../5-EagleEye-training). The dataset gets stored at [4-Create-dataset/dataset.npz](./dataset.npz). The dataset consists of a list of compressed numpy arrays. 

You can investigate the dataset as follows:

```
import numpy as np
dataset = np.load('4-Create-dataset/dataset.npz', allow_pickle=True)
print(f"What's included in the dataset: {list(dataset.keys())}.")
x_train = dataset['x_train']
print(f"Shape of training data: {x_train.shape}.")
print(f"List of all features, after one-hot encoding: {dataset['features_names_after_preprocessing']}.")
```

The training data is of shape (`number of windows`, `sequence length`, `number of security features after OHE`). In the paper, we use sequences of 200 events, where each event has roughly 200 features. Note that the final dataset will have significantly more features than the number of security features. This is due to one-hot encoding, where each categorical feature gets mapped to a list equal in size to the number categories.

The training labels are of shape (`number of windows`, 2), where each label is one-hot encoded (either `[0,1]` or `[1,0]`).

## Create your own dataset

### Bring your own provenance graphs
You can create your own dataset by applying the scripts here to your own process provenance graphs. Make sure to include enough graphs, so that the training data covers a diverse set of application types. Additionally, you'll need to extract *security* features from your graphs. You can use step [2-Security-features](../2-Security-features) as a reference for this. Make sure all your security features are in of numerical, categorical, or boolean nature. Also, make sure your graphs are in JSON tree format.

### Configure the dataset script

The dataset creation requires several hyper parameters. You can see those hyper parameters in the [4-Create-dataset/main.py](main.py) script:
- `WINDOW_SIZE`: The length of the behavior sequences. EagleEye uses windows of 200 events.
- `OVERLAP_FACTOR`: The amount of overlap a window has, on average. A value of 2 means every event gets into two windows, on average.
- `GRAPHS_BENIGN` and `GRAPHS_MALICIOUS`: The folders which contain your large set of provenance graphs.
- `FEATURE_SCHEMA_CSV`: A file path to your security feature schema. This file should contain a list of the security features you are using.

## Data processing pipeline - details

This data processing step contains performs the following transformations:
- [4-Create-dataset/graphs_to_windows.py](./graphs_to_windows.py)
  - Turn each graph into a linear sequence of events
  - Turn each event into a security feature vector (of fix size)
  - Slice the behavior sequences into fixed-size windows
- [4-Create-dataset/windows_to_dataset.py](./windows_to_dataset.py)
  - Continue with the windows from the previous step
  - One-hot encode categorical features
  - Normalize numerical features (turn into standard normal distribution)
  - Split data into two parts: a `train` dataset and a `validation` dataset
  - Store final result as compressed numpy array

