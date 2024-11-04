import numpy as np
import torch

from pathlib import Path
import networkx as nx
from networkx.readwrite import json_graph
import json
from sentence_transformers import SentenceTransformer
import os
import shutil

from tqdm import tqdm

from models import AutoEncoder


# Compute command line embeddings
#
# Use a trained sentence transformer plus an auto encoder to turn command line Strings into a 16-dimensional vector.
#
# Make sure the trained autoencoder is available at the path defined by the variable `AUTOENCODER_PATH`.


AUTOENCODER_PATH = 'weights/ae'
st_model = SentenceTransformer('all-MiniLM-L6-v2')
autoencoder = AutoEncoder.load(AUTOENCODER_PATH)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def embed_cmd(cmdline) -> np.ndarray:
    """
    Embed one preprocessed cmdline (child + parent process) and return result as list of 16 embedding values.

    So one String gets embedded into one 16-dimensional float vector.
    Use the pretrained sentence transformer for this purpose.

    Arguments:
        - cmdline: The preprocessed command line string
    
    Return:
        - A numpy-array
    """
    autoencoder.eval()
    with torch.no_grad():
        emb = st_model.encode([cmdline], convert_to_tensor=True, device=device.type)
        compression_embeddings = autoencoder.encode(emb)

    return compression_embeddings[0].cpu().numpy()


def enrich_graph(graph: nx.DiGraph) -> nx.DiGraph:
    """
    Add cmdline embeddings to a BE graph.
    Arguments:
        - graph: The DiGraph with the process tree
    """
    # find nodes of type "si_create_process"
    node_ids_and_cmdlines = nx.get_node_attributes(graph, "preprocessed_cmdline_for_transformer")
    # compute cmdline embeddings
    node_ids_and_cmdline_embeddings = [(node_id, embed_cmd(cmd)) for (node_id, cmd) in node_ids_and_cmdlines.items()]
    # add embedding to graph
    for node_id, embedding in node_ids_and_cmdline_embeddings:
        for i in range(0, 16):
            graph.nodes[node_id]["encoding_cmdline_embedding_{}".format(i)] = float(embedding[i])
    return graph


def enrich_folder_graphs(folder_source: str, folder_target: str) -> None:
    """
    Perform command line embedding and enrich all graphs in a folder.

    Search for graphs recursively within the source folder.
    Arguments:
        - folder_source: The folder with the source graphs
        - folder_target: The folder where the enriched graphs will be stored
    """
    if os.path.exists(folder_target):
        shutil.rmtree(folder_target)
    os.makedirs(folder_target)
    # list all graph files
    files = list(Path(folder_source).rglob("*.json"))
    # for each graph ...
    print(f"Enriching {len(files)} graphs")
    for file in tqdm(files):
        # read in graph
        with open(file, 'r') as f:
            json_data = json.load(f)
        graph = json_graph.tree_graph(json_data)
        # compute cmdline embeddings
        graph = enrich_graph(graph)
        # persist new graph
        head = [idx for idx, degree in graph.in_degree() if degree == 0][0]
        json_data = json_graph.tree_data(graph, root=head)
        with open(folder_target + "/" + os.path.basename(file), 'w') as f:
            json.dump(json_data, f)


if __name__ == '__main__':
    # Assign the input and output folders. Ensure that the input folders
    # here contain the process provenance graphs to be enriched.
    # it is recommended to use the same folder structure for the data
    # employed in training the Autoencoder as described in preprocessing.py
    input_folders = [
        "data/graphs/malicious/test/",
    ]
    output_folders = [
        "data/enriched/malicious/test/",
    ]
    auto_encoder = AutoEncoder.load(AUTOENCODER_PATH)
    for in_folder, out_folder in zip(input_folders, output_folders):
        enrich_folder_graphs(in_folder, out_folder)


