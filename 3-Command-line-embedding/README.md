# Embed command line Strings from BE graphs

Embed commandlines with sentence transformer, then train an autoencoder to reduce the dimensionality of the embeddings.

## Run

Processing steps:
1. Place the BE graphs into the `data/ae/graphs` folder, separated into `benign` and `malicious` subfolders which further contain train/val/test splits.
2. Preprocess and vectorize the command line Strings: `main.preprocess_data()` from the BE graphs.
3. Train the Autoencoder: `main.train_autoencoder()`, the weights are stored in `weights/ae/pytorch_model.pt`.
4. Enrich the BE graphs with the embeddings: `graph_embedding.py`.
