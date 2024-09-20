"""An implementation of a transformer encoder.

This code allows you to train a transformer for classification, from scratch.
The input for the transformer are sequences, where each element in the sequence is a 1-d vector.

Reference: https://www.tensorflow.org/text/tutorials/transformer
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from tensorflow.keras.regularizers import l2

class PositionalEmbedding(tf.keras.layers.Layer):
    """
    Do both embedding (dimension reduction) and add a positional offset.
    """

    def _positional_encoding(self, length: int, depth: int) -> tf.Tensor:
        """
        Compute a positional encoding, as sinusoidal function.
        """
        depth = depth/2

        positions = np.arange(length)[:, np.newaxis]
        depths = np.arange(depth)[np.newaxis, :]/depth

        angle_rates = 1 / (10000**depths)
        angle_rads = positions * angle_rates

        pos_encoding = np.concatenate(
            [np.sin(angle_rads), np.cos(angle_rads)],
            axis=-1) 

        return tf.cast(pos_encoding, dtype=tf.float32)

    def __init__(self, d_model: int, positional_encoding_period: int):
        super().__init__()
        self.d_model = d_model
        self.embedding = tf.keras.layers.Dense(d_model) 
        self.pos_encoding = self._positional_encoding(length=positional_encoding_period, depth=d_model)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """The forward path of this TF layer."""
        length = tf.shape(x)[1]
        x = self.embedding(x)
        # This factor sets the relative scale of the embedding and positonal_encoding.
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x

class SelfAttention(tf.keras.layers.Layer):
    """
    Multi head attention + layer normalization + residual connection.
    """
    def __init__(self, num_heads: int, key_dim: int, dropout_rate: int):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout_rate)
        self.add = tf.keras.layers.Add()
        self.layernorm = tf.keras.layers.LayerNormalization()

    def call(self, x: tf.Tensor):
        """The forward path of this TF layer."""
        scores = self.mha(x,x,x)
        # apply normalization before residual connection, as proposed in T5 paper
        norm = self.layernorm(scores)
        sum = self.add([x, norm])
        return sum

class FeedForward(tf.keras.layers.Layer):
    """
    Feed forward neural network + layer normalization + residual connection.
    """
    def __init__(self, d_model, dff, dropout_rate):
        """
        :param d_model: dimension of token representations
        :param dff: token dimension after first dense layer
        """
        super().__init__()
        self.nn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model),
            tf.keras.layers.Dropout(dropout_rate)
        ])
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """The forward path of this TF layer."""
        nn_output = self.nn(x)
        # apply normalization before residual connection, as proposed in T5 paper
        norm = self.layer_norm(nn_output)
        res = self.add([norm, x])
        return res

class EncoderLayer(tf.keras.layers.Layer):
    """
    A complete encoder, including everything from self attention to feed forward neural network.
    """
    def __init__(self, d_model, num_heads, key_dim, dff, dropout_rate):
        super().__init__()
        self.self_attention = SelfAttention(
                num_heads=num_heads,
                key_dim=key_dim,
                dropout_rate=dropout_rate)
        self.ffn = FeedForward(d_model, dff, dropout_rate)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """The forward path of this TF layer."""
        x = self.self_attention(x)
        x = self.ffn(x)
        return x

class Transformer(tf.keras.Model):
    """
    A transformer encoder classifier
    """
    def __init__(self, num_layers, d_model, num_heads, key_dim, dff,
                 dropout_rate, positional_encoding_period, regularization):
        super().__init__()

        self.num_layers = num_layers
        self.regularization = regularization
        self.pos_embedding = PositionalEmbedding(d_model=d_model, positional_encoding_period=positional_encoding_period)
        self.dropout_embedding = tf.keras.layers.Dropout(dropout_rate)
        self.encoders = [
                EncoderLayer(d_model=d_model,
                        num_heads=num_heads,
                        key_dim=key_dim,
                        dff=dff,
                        dropout_rate=dropout_rate)
                for _ in range(num_layers)]
        self.final_ff = tf.keras.layers.Dense(dff, activation='relu', kernel_regularizer=l2(regularization))
        self.dropout_final_ff = tf.keras.layers.Dropout(dropout_rate)
        self.final_ff_softmax = tf.keras.layers.Dense(2, activation = 'softmax')

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """The forward path of this TF model
        :param x: has shape (batch size, sequence length, number of features)
        """
        x = self.pos_embedding(x)
        x = self.dropout_embedding(x)
        for i in range(self.num_layers):
            x = self.encoders[i](x)
        cls_representation = x[:,0,:]
        nn = self.final_ff(cls_representation)
        nn = self.dropout_final_ff(nn)
        nn = self.final_ff_softmax(nn)
        return nn

class TransformerTraining:
    """Use this class to train a transformer."""
    CLS = -1
  
    def _add_cls_token(self, x: np.ndarray) -> np.ndarray:
        """
        For each window, add an additional CLS event to the start.
        """
        cls_event = np.repeat(self.CLS, repeats = x.shape[2]).reshape(1,-1)
        x = np.insert(x, 0, cls_event, axis=1)
        return x
  
    def _plot_training(self, history) -> None:
        """Show the training progress."""
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Transformer training progress')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='lower right')
        plt.grid(visible=True)
        plt.show()
    
    def train(self, x_train, y_train, x_val, y_val, num_layers, d_model, num_heads, key_dim, dff, dropout_rate, positional_encoding_period,
              regularization, learning_rate, patience, checkpoint_folder, epochs, batch_size) -> None:
        """Perform the transformer encoder training, from scratch, on sequence classification.

        After the training, show the training progress as plot.

        :param x_train: The training data of shape (number of samples, sequence length, number of features per behavior event)
        :param y_train: The training labels of shape (number of samples, 2), where the data is one-hot encoded
        :param x_val:
        :param y_val:
        :param num_layers: The encoder stack size
        :param d_model: The dimension of tokens inside the transformer model
        :param num_heads: The number of parallel attention heads for multi-head attention
        :param key_dim: The dimension of key tokens for the attention mechanism
        :param dff: The dimension of tokens inside the feed forward neural network
        :param dropout_rate: The dropout for the FF neural network at the encoder top
        :param positional_encoding_period: The period of the positional encoding, which should be longer than the input sequences
        :param regularization: The regularization in the final dense layers at the top of the encoder stack
        :param learning_rate: the training learning rate
        :param patience: number of epochs to wait for a training improvement, before an early stop
        :param checkpoint_folder: Store the best model under this relative file path
        :param epochs: number of training epochs
        :param batch_size: training batch size
        """
        
        x_train = self._add_cls_token(x_train)
        x_val = self._add_cls_token(x_val)

        self.transformer = Transformer(
                num_layers = num_layers,
                d_model = d_model,
                num_heads = num_heads,
                key_dim = key_dim,
                dff = dff,
                dropout_rate = dropout_rate,
                positional_encoding_period = positional_encoding_period,
                regularization = regularization)

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        self.transformer.compile(
                loss = 'categorical_crossentropy',
                optimizer = optimizer,
                metrics = ['accuracy'])

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss',
                patience = patience,
                mode = 'min')

        checkpoint = tf.keras.callbacks.ModelCheckpoint(
                filepath = f"{checkpoint_folder}/checkpoint.model.keras",
                monitor='val_loss',
                verbose=0,
                save_best_only=True,
                mode='min')

        print("Starting transformer training ...")
        history = self.transformer.fit(
                x_train, y_train, epochs = epochs,
                validation_data = (x_val, y_val),
                callbacks = [early_stopping, checkpoint],
                batch_size = batch_size)
        
        self._plot_training(history)

    def perform_evaluation(self, x_val: np.ndarray, y_val: np.ndarray) -> None:
        """Evaluate the trained ML model on the validation dataset.
        
        Draw the receiver operator characterstic (ROC) curve, and compute the classification accuracy.
        """
        print("Run inference on validation dataset ...")
        x_val = self._add_cls_token(x_val)
        y_prediction = self.transformer.predict(x_val)
        y_prediction = [y[0] for y in y_prediction]
        y_label = [y[0] for y in y_val]
        y_prediction_binary = np.array(y_prediction) > 0.5
        accuracy = sum(y_label == y_prediction_binary) / len(y_val)
        print(f"Validation accuracy: {accuracy}.")
        fpr, tpr, _ = metrics.roc_curve(y_label, y_prediction)
        roc_auc = metrics.auc(fpr, tpr)
        display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                          estimator_name='Transformer')
        display.plot()
        plt.title('Transformer on validation dataset, MALICIOUS=1')
        plt.grid(visible=True)
        plt.show()
