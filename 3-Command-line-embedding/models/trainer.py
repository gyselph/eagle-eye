import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt, animation
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from .nn import AutoEncoder


class AutoEncoderTrainer:
    """
    AutoEncoder model trainer for command line embeddings compression
    """

    def __init__(self, embed_dim: int, encoder_dim: int, x_train: pd.DataFrame, ben_validation: pd.DataFrame,
                 mal_validation: pd.DataFrame, lr=0.001, epochs=7, batch_size=16, num_workers=2):
        """
        param: embed_dim: int: Dimension of the input embeddings
        param: encoder_dim: int: Dimension of the encoder output
        param: x_train: pd.DataFrame: Training data with shape (n_samples, embed_dim)
        param: ben_validation: pd.DataFrame: Validation data with shape (n_samples, embed_dim)
        param: mal_validation: pd.DataFrame: Validation data with shape (n_samples, embed_dim)
        param: lr: float: Learning rate for the optimizer
        param: epochs: int: Number of epochs to train the model
        param: batch_size: int: Batch size for training
        param: num_workers: int: Number of workers for the DataLoader
        """
        self.embed_dim = embed_dim
        self.encoder_dim = encoder_dim
        self.device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        self.model = AutoEncoder(embed_dim, encoder_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.L1Loss(reduction='none')
        self.x_train = torch.FloatTensor(x_train.to_numpy())
        self.ben_validation = torch.FloatTensor(ben_validation.to_numpy())
        self.mal_validation = torch.FloatTensor(mal_validation.to_numpy())
        self.train_dataset = TensorDataset(self.x_train)
        self.x_val = TensorDataset(torch.concat([self.ben_validation, self.mal_validation], dim=0))
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train(self):
        # get the training dataloader
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
        self.model.train()
        print_intervals = 1
        iters = 0
        losses = []
        epoch_losses = []
        ben_val_losses = []
        mal_val_losses = []
        val_epoch_losses = []

        # training loop
        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0
            for batch, (x_batch,) in enumerate(train_loader):
                x_batch = x_batch.to(self.device)

                x_pred = self.model(x_batch)
                loss = self.criterion(x_pred, x_batch).mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                losses.append(loss.item())
                iters += 1

            epoch_loss /= len(train_loader)
            epoch_losses.append(epoch_loss)

            # run validation every print_intervals epochs
            if epoch % print_intervals == 0:
                self.model.eval()
                with torch.no_grad():
                    ben_loader = DataLoader(TensorDataset(self.ben_validation), batch_size=self.batch_size,
                                            shuffle=True)
                    ben_losses = []
                    for (batch,) in ben_loader:
                        batch = batch.to(self.device)
                        val_pred = self.model(batch)
                        ben_losses += self.criterion(val_pred, batch).mean(dim=-1).numpy().tolist()
                    ben_val_losses.append(ben_losses)

                    mal_loader = DataLoader(TensorDataset(self.mal_validation), batch_size=self.batch_size,
                                            shuffle=True)
                    mal_losses = []
                    for (batch,) in mal_loader:
                        batch = batch.to(self.device)
                        val_pred = self.model(batch)
                        mal_losses += self.criterion(val_pred, batch).mean(dim=-1).numpy().tolist()
                    mal_val_losses.append(mal_losses)
                    val_epoch_losses.append((np.mean(ben_losses) + np.mean(mal_losses)) / 2)
                self.model.train()
                print(f'[Epoch {epoch}]:\t Train Loss: {epoch_loss} \t Benign Validation Loss:'
                      f' {np.mean(ben_losses)} \t Malicious Validation Loss: {np.mean(mal_losses)}')

        # plot the training and validation losses
        train_loss = np.array(epoch_losses).reshape(self.epochs, -1).mean(axis=1)
        val_loss = np.array(val_epoch_losses).reshape(self.epochs, -1).mean(axis=1)
        plt.plot(train_loss, label='Train loss')
        plt.plot(val_loss, label='Validation loss')
        plt.title('Training loss')
        plt.ylabel('Loss')
        plt.xlabel('Iterations')
        plt.legend()
        plt.show()

        # plot the validation loss distribution
        ben_val_losses = np.concatenate(ben_val_losses)
        mal_val_losses = np.concatenate(mal_val_losses)
        plt.hist(ben_val_losses, label='Benign', bins=50)
        plt.hist(mal_val_losses, label='Malicious', bins=50)
        plt.title('Validation Reconstruction Loss distribution')
        plt.ylabel('Frequency')
        plt.xlabel('Reconstruction loss')
        plt.legend()
        plt.show()

        # save the model
        self.model.save('weights/ae/')

        return epoch_losses, ben_val_losses, mal_val_losses

    def test(self, ben_test: pd.DataFrame, mal_test: pd.DataFrame, batch_size=16):
        """
        Test the trained model on the test data.
        param: ben_test: pd.DataFrame: Test data for benign samples
        param: mal_test: pd.DataFrame: Test data for malicious samples
        param: batch_size: int: Batch size for testing
        """
        ben_test = torch.FloatTensor(ben_test.to_numpy())
        ben_predicted = []
        ben_losses = []
        ben_feature_losses = []
        ben_loader = DataLoader(TensorDataset(ben_test), batch_size=batch_size, shuffle=True)
        for (batch,) in tqdm(ben_loader):
            batch = batch.to(self.device)
            x_pred = self.predict(batch)
            ben_losses += self.criterion(x_pred, batch).mean(dim=1).cpu().numpy().tolist()
            ben_feature_losses.append(self.criterion(x_pred, batch).mean(dim=0).cpu().numpy().tolist())
            ben_predicted.append(x_pred.cpu().numpy())
        ben_predicted = np.concatenate(ben_predicted, axis=0)

        mal_test = torch.FloatTensor(mal_test.to_numpy())
        mal_predicted = []
        mal_losses = []
        mal_feature_losses = []
        mal_loader = DataLoader(TensorDataset(mal_test), batch_size=batch_size, shuffle=True)
        for (batch,) in tqdm(mal_loader):
            batch = batch.to(self.device)
            x_pred = self.predict(batch)
            mal_losses += self.criterion(x_pred, batch).mean(dim=1).cpu().numpy().tolist()
            mal_feature_losses.append(self.criterion(x_pred, batch).mean(dim=0).cpu().numpy().tolist())
            mal_predicted.append(x_pred.cpu().numpy())
        mal_predicted = np.concatenate(mal_predicted, axis=0)

        return ben_losses, ben_predicted, mal_losses, mal_predicted

    def predict(self, embeddings: torch.FloatTensor) -> torch.FloatTensor:
        """
        Get the reconstructed embeddings for the input embeddings.
        param: embeddings: torch.FloatTensor: Input embeddings with shape
        (n_samples, embed_dim) obtained from the SentenceTransformer model
        """
        self.model.eval()
        with torch.no_grad():
            x_pred = self.model(embeddings)
            return x_pred
