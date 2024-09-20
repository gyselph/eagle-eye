# Train a transformer from scratch, on behavior event sequences

This folder demos how to train an encoder-only transformer on behavior sequences, from scratch.

For EagleEye, we train the transformer on the binary classification task. Each behavior sequence needs to be categorized into being either *malicious* or *benign*.

## Run code

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
python ./1-EagleEye-training/main.py
```

## The dataset

At this point, the script uses a random dataset. This should be replaced by a real process behavior dataset.

The training data is of shape (`number of samples`, `sequence length`, `number of features per behavior event`). In the paper, we use sequences of 200 events, where each event has roughly 200 features.

The training labels are of shape (`number of samples`, 2), where each sampleis one-hot encoded (either `[0,1]` or `[1,0]`).
        