import os.path
import time
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sentence_transformers import SentenceTransformer

from preprocessing import prepare_data
from models import AutoEncoder, AutoEncoderTrainer

MAX_LEN = 1000
np.random.seed(100)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def preprocess_data(force: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Prepare the data for training. If the data is already preprocessed, it will load the preprocessed data.

    The preparation process takes command line strings fom provenance graphs and
    transforms them to high dimensional embedding vectors stored as csv files

    param: force: bool: If True, the data will be re-preprocessed
    """
    if not force and os.path.exists('data/preprocessed/benign') and os.path.exists('data/preprocessed/malicious'):
        ben_train = pd.read_csv('data/preprocessed/benign/train_features.csv', index_col=0)
        ben_val = pd.read_csv('data/preprocessed/benign/val_features.csv', index_col=0)
        ben_test = pd.read_csv('data/preprocessed/benign/test_features.csv', index_col=0)
        mal_train = pd.read_csv('data/preprocessed/malicious/train_features.csv', index_col=0)
        mal_val = pd.read_csv('data/preprocessed/malicious/val_features.csv', index_col=0)
        mal_test = pd.read_csv('data/preprocessed/malicious/test_features.csv', index_col=0)
        return ben_train, ben_val, ben_test, mal_train, mal_val, mal_test
    ben_train, ben_val, ben_test, mal_train, mal_val, mal_test = prepare_data()
    return ben_train, ben_val, ben_test, mal_train, mal_val, mal_test


def train_autoencoder(
        ben_train: pd.DataFrame,
        ben_val: pd.DataFrame,
        ben_test: pd.DataFrame,
        mal_train: pd.DataFrame,
        mal_val: pd.DataFrame,
        mal_test: pd.DataFrame,
        epochs: int = 100
) -> AutoEncoder:
    """
    Train the command line autoencoder model and test it on the test data.
    As input, the autoencoder takes the high dimensional embedding vectors of the command line stings
    obtained in the preprocessing step and compresses them 16-dimensional vectors

    param: ben_train: pd.DataFrame: Benign training data
    param: ben_val: pd.DataFrame: Benign validation data
    param: ben_test: pd.DataFrame: Benign test data
    param: mal_train: pd.DataFrame: Malicious training data
    param: mal_val: pd.DataFrame: Malicious validation data
    param: mal_test: pd.DataFrame: Malicious test data
    param: epochs: int: Number of epochs to train the model
    """
    begin = time.time()
    x_train = pd.concat([ben_train, mal_train])

    torch.manual_seed(100)
    trainer = AutoEncoderTrainer(
        embed_dim=len(x_train.columns),
        encoder_dim=16,
        x_train=x_train,
        ben_validation=ben_val,
        mal_validation=mal_val,
        epochs=epochs
    )
    trainer.train()

    # test the model
    ben_losses, ben_predicted, mal_losses, mal_predicted = trainer.test(ben_test, mal_test)

    # plot the reconstruction loss distribution on the test data
    # highlighting the benign and malicious samples separately
    plt.hist(ben_losses, label='Benign', bins=50)
    plt.hist(mal_losses, label='Malicious', bins=50)
    plt.title('Test Reconstruction Loss distribution')
    plt.ylabel('Frequency')
    plt.xlabel('Reconstruction loss')
    plt.legend()
    plt.show()
    print(f"TIME ELAPSED: {(time.time() - begin) / 60} minutes or {time.time() - begin} seconds")

    return trainer.model


def encode_cmdlines(cmdlines:list[str], model_path: str) -> torch.FloatTensor:
    """
    Encode the command lines using the trained model.
    First the command lines are encoded using the SentenceTransformer model,
    then the embeddings are compressed to size of 16 using the trained autoencoder model.
    param: cmdlines: list[str]: List of command lines to encode
    param: model_path: str: Path to the trained autoencoder model
    """
    st_model = SentenceTransformer('all-MiniLM-L6-v2')
    compression_model = AutoEncoder.load(model_path)
    print('Computing embeddings')
    compression_model.eval()
    with torch.no_grad():
        emb = st_model.encode(cmdlines, convert_to_tensor=True, device=device.type)
        compression_embeddings = compression_model.encode(emb)

    return compression_embeddings


if __name__ == '__main__':
    datasets = preprocess_data(force=False)
    train_autoencoder(*datasets, epochs=2)
    embeddings = encode_cmdlines(['ls', 'cd | pwd', 'pwd | kill'], 'weights/ae')