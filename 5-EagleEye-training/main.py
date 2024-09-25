"""Main script to perform the transformer training.

Fit an encoder-transformer on sequences, for binary classification.
We use the previously generated dataset from step '4-Create-dataset'.
"""

import numpy as np

from train_transformer import TransformerTraining

# TODO: set these hyper parameters to reasonable values, or perform a hyper parameter search

# number of training epochs
EPOCHS = 5
# number of epochs to wait for a training improvement, before an early stop
PATIENCE = 3
# the dimesion of tokens within the transformer
D_MODEL = 30
# number of attention heads for multi-head self attention
ATTENTION_HEADS = 2
# the transformer encoder stack size
NUM_ENCODERS = 2
# training batch size
BATCH_SIZE = 5
# length of behavior sequences
WINDOW_SIZE = 50
# the period after which the positional embedding gets repeated
POSITIONAL_ENCODING_PERIOD = 5 * WINDOW_SIZE
# the training learning rate
LR = 0.0001
# the dropout rate throughout the transformer
DROPOUT = 0.1
# the regularization in the final dense layers at the top of the encoder stack
REGULARIZATION = 0.01
# the dimension of keys for the attention mechanism
KEY_DIM = D_MODEL
# the internal dimension in the feed-forward neural network, within each encoder
DIM_FF = 4 * D_MODEL
# Store the best model under this relative file path
CHECKPOINT_FOLDER = "."
# File path to the dataset we train on - we use the result of the previous processing step
DATASET = './4-Create-dataset/dataset.npz'

# TODO: create your own large dataset of >1k graphs
def load_dataset(file_path: str):
    """Load the dataset.
    
    :param file_path: The relative path to the file with the numpy dataset.
    """
    dataset = np.load(file_path, allow_pickle=True)
    return dataset['x_train'].astype(float), dataset['y_train'], dataset['x_val'].astype(float), dataset['y_val']

# load the dataset
x_train, y_train, x_val, y_val = load_dataset(DATASET)
# perform the transformer training on the training data
transformer_training = TransformerTraining()
transformer_training.train(
        x_train = x_train,
        y_train = y_train,
        x_val = x_val,
        y_val = y_val,
        num_layers = NUM_ENCODERS,
        d_model = D_MODEL,
        num_heads = ATTENTION_HEADS,
        key_dim = KEY_DIM,
        dff = DIM_FF,
        dropout_rate = DROPOUT,
        positional_encoding_period = POSITIONAL_ENCODING_PERIOD,
        regularization = REGULARIZATION,
        learning_rate = LR,
        patience = PATIENCE,
        checkpoint_folder = CHECKPOINT_FOLDER,
        epochs = EPOCHS,
        batch_size = BATCH_SIZE)
# Evaluate the trained model on the validation data.
# The results don't look great? That's to be expected when you train on such a small training sample!
transformer_training.perform_evaluation(x_val, y_val)
