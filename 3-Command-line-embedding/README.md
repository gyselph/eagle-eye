# Embed command line Strings from BE graphs

Embed commandlines with sentence transformer, then train an autoencoder to reduce the dimensionality of the embeddings.

## Run
The working directory for running scripts in this section is `3-Command-line-embedding`. Hence, run `cd 3-Command-line-embedding` before any scripts.

Processing steps:
1. To benefit from the command line context enrichment, the process provenance graphs should contain atleast one command line string. The command line string is placed in an 
 attribute called `preprocessed_cmdline_for_transformer` where each string is a concatenation of the parent and child commands using the vertical bar (`|`) separator.
 For example, if the parent process command is `cd folder` and the child process command is `rm -rf *.txt` then the value of `preprocessed_cmdline_for_transformer` for the child
 process would be `cd folder | rm -rf *.txt`. Adding the parent command provides extra context to the command.
2. Place the process provenance graphs into the `data/graphs` folder, separated into `benign` and `malicious` subfolders which further contain train/val/test splits.
3. Preprocess and vectorize the command line Strings: `main.preprocess_data()` from the graphs. This step uses a pre-trained sentence transformer "all_MiniLM_L6_v2", to turn the command-line strings into embeddings of 340 dimensions.
4. Train the Autoencoder: `main.train_autoencoder()`, the weights are stored in `weights/ae/pytorch_model.pt`. The autoencoder reduces the dimension of the original embeddings to 16.
5. Enrich the graphs with the embeddings: `graph_embedding.py`.
