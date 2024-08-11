import pdb

import pandas as pd
import numpy as np
import math
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm


class BaselineModel:

    def __init__(self, group_cols: list, col_to_shift: str):
        """
        Create a new BaselineModel
        :param group_cols: column names to group by when shifting
        :param col_to_shift: the column to shift to create a naive y[i] = y[i-1] prediction
        """
        self.group_cols = group_cols
        self.col_to_shift = col_to_shift

    def add_shifted_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add a shifted column to the df
        :param df: the data to shift
        :return: the same DataFrame with a shifted column added
        """
        sort_cols = ['season'] + self.group_cols

        # Make sure the DataFrame is sorted appropriately
        df_sorted = df.sort_values(by=sort_cols)

        # Perform the shift operation within each group
        df_sorted['shifted_' + self.col_to_shift] = df_sorted.groupby(self.group_cols)[self.col_to_shift].shift(1)

        # Restore the original order of the DataFrame
        df_sorted = df_sorted.sort_index()

        return df_sorted

    def predict(self, df) -> np.ndarray:
        """
        Get the predictions from the dataframe by shifting 1 season
        :param df: the data to predict on
        :return: the shifted data
        """
        # return self.add_shifted_column(df)['shifted_'+self.col_to_shift].values
        return df['total_points']


class BaselineModel2(BaselineModel):
    """
    Another baseline with uses the average over the last 2 years instead of just last year
    """

    def __init__(self, group_cols: list, col_to_shift: str):
        """
        Create a new BaselineModel
        :param group_cols: column names to group by when shifting
        :param col_to_shift: the column to shift to create a naive y[i] = y[i-1] prediction
        """
        super().__init__(group_cols, col_to_shift)

    def add_shifted_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add a shifted column to the df
        :param df: the data to shift
        :return: the same DataFrame with a shifted column added
        """
        sort_cols = ['season'] + self.group_cols

        # Make sure the DataFrame is sorted appropriately
        df_sorted = df.sort_values(by=sort_cols)

        # Perform the shift operation within each group
        df_sorted['shifted_' + self.col_to_shift] = df_sorted.groupby(self.group_cols)[self.col_to_shift].shift(1)

        # Restore the original order of the DataFrame
        df_sorted = df_sorted.sort_index()

        return df_sorted

    def calc_mean(self, num1: float, num2: float) -> float:
        """
        calculate the mean of 2 numbers with handling for nan
        :param num1: the first number
        :param num2: the second number
        :return: the mean of the 2 numbers
        """
        # Check if both numbers are NaN
        if math.isnan(num1) and math.isnan(num2):
            return math.nan

        # Check if one number is NaN
        if math.isnan(num1):
            return num2
        if math.isnan(num2):
            return num1

        # Calculate and return the mean of the two numbers
        return (num1 + num2) / 2

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Get the predictions from the dataframe by shifting 1 season
        :param df: the data to predict on
        :return: the shifted and averaged data
        """
        df = self.add_shifted_column(df)
        og_col_to_shift = self.col_to_shift
        self.col_to_shift = 'shifted_' + og_col_to_shift
        df = self.add_shifted_column(df)
        pred = df.apply(lambda x: self.calc_mean(x[self.col_to_shift], x['shifted_' + self.col_to_shift]))
        # reset col_to_shift for another potential run
        self.col_to_shift = og_col_to_shift
        return pred


class NeuralNetRegressor(torch.nn.Module):

    def __init__(self, activation_function: str, lr: float, layers: list[int], epochs: int, gamma: float = 0.9):
        """
        Create a new NeuralNetRegressor
        :param activation_function: the activation function for all layers
        :param lr: learning rate for training
        :param layers: list of nodes per layer
        :param epochs: number of epochs to train
        :param gamma: gamma for learning rate scheduler
        """
        super().__init__()
        self.seed = 42
        torch.manual_seed(self.seed)
        activations = {
            'relu': torch.nn.ReLU(),
            'sigmoid': torch.nn.Sigmoid(),
            'tanh': torch.nn.Tanh(),
            'leaky_relu': torch.nn.LeakyReLU(),
        }
        self.gamma = gamma
        self.activation_function = activation_function
        self.lr = lr
        self.layers = layers
        self.epochs = epochs
        self.model = torch.nn.Sequential()
        self.criterion = torch.nn.MSELoss()
        self.input_dimension = 29  # number of features in dataset
        self.model.add_module('input', torch.nn.Linear(self.input_dimension, self.layers[0]))
        for i in range(len(self.layers) - 1):
            self.model.add_module(f'layer{i}', torch.nn.Linear(self.layers[i], self.layers[i + 1]))
            self.model.add_module(f"{self.activation_function}_{i}", activations[self.activation_function])
        self.model.add_module('output', torch.nn.Linear(self.layers[-1], 1))

    def forward(self, data):
        """
        Forward pass for data through the model
        :param data: the data to process
        :return: the output of the model
        """
        return self.model(data)

    def fit(self, X_train, y_train, batch_size=32):
        # Convert the training data to PyTorch tensors
        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)

        # Create a DataLoader for batching
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Set the optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = ExponentialLR(self.optimizer, gamma=self.gamma)

        # Training loop
        for epoch in tqdm(range(self.epochs), leave=True):
            for X_batch, y_batch in train_loader:
                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass
                y_pred = self.model(X_batch)
                loss = self.criterion(y_pred, y_batch.unsqueeze(1))  # unsqueeze adds a dimension for the batch size

                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()
            scheduler.step()

            # Print the loss for every epoch
            # print(f'Epoch [{epoch + 1}/{self.epochs}], Loss: {loss.item():.4f}')

        return self

    def predict(self, X_test):
        # Convert the test data to PyTorch tensors
        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)

        # Make predictions
        with torch.no_grad():
            y_pred = self.model(X_test_tensor)

        return y_pred.numpy()


class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, epochs: int=10, lr: float = 0.01, gamma: float = 0.9, dropout: float=0):
        super(LSTMRegressor, self).__init__()
        self.seed = 42
        torch.manual_seed(self.seed)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.epochs = epochs
        self.lr = lr
        self.gamma = gamma
        self.dropout = dropout

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)

        # Fully connected layer for regression output
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, lengths):
        # Pack padded sequence to handle varying lengths
        packed_input = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        # Pass through LSTM
        packed_output, (hn, cn) = self.lstm(packed_input)

        # Unpack sequence to get the output
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # Use the last hidden state of the LSTM as the feature for regression
        # Gather the last relevant output for each sequence
        idx = (lengths - 1).view(-1, 1).expand(len(lengths), self.hidden_size)
        idx = idx.unsqueeze(1)
        last_output = output.gather(1, idx).squeeze(1)

        # Pass through the fully connected layer to get the final output
        out = self.fc(last_output)

        return out



class LSTMRegressorWrapper:

    def __init__(self, input_size, hidden_size, num_layers, output_size, epochs: int=10, lr: float = 0.01, gamma: float = 0.9, dropout: float=0):
        self.model = LSTMRegressor(input_size, hidden_size, num_layers, output_size, dropout)
        self.epochs = epochs
        self.lr = lr
        self.gamma = gamma
        self.dropout = dropout
        self.criterion = torch.nn.MSELoss()

    @staticmethod
    def dataframe_to_dataloader(df, groupby_columns, sort_column, target_column, batch_size=32,
                                shuffle=True):
        """
        Converts a pandas DataFrame into a PyTorch DataLoader for an LSTM.

        Args:
            df (pd.DataFrame): The input dataframe.
            groupby_columns (list): List of columns to group by.
            sort_column (str): The column to sort by within each group.
            target_column (str): The column to be used as the target for regression.
            batch_size (int): Number of samples per batch.
            shuffle (bool): Whether to shuffle the data after creating sequences.

        Returns:
            DataLoader: A PyTorch DataLoader ready for consumption by an LSTM.
        """
        sequences = []
        targets = []
        lengths = []

        feature_columns = [x for x in df.columns if x not in [target_column, sort_column] + groupby_columns]

        # Group by specified columns and sort within each group
        grouped = df.groupby(groupby_columns)

        for _, group in grouped:
            # Sort the group by the specified column
            group = group.sort_values(by=sort_column)

            # Extract features and target sequences
            feature_sequence = torch.tensor(group[feature_columns].values, dtype=torch.float32)
            target_value = torch.tensor(group[target_column].values[-1],
                                        dtype=torch.float32)  # Use the last target value

            sequences.append(feature_sequence)
            targets.append(target_value)
            lengths.append(len(feature_sequence))

        # Pad sequences to the same length
        padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0.0)
        lengths = torch.tensor(lengths, dtype=torch.long)
        targets = torch.tensor(targets, dtype=torch.float32)

        # Create a TensorDataset
        dataset = TensorDataset(padded_sequences, lengths, targets)

        # Create a DataLoader
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        return dataloader

    def fit(self, train_loader, num_epochs=10, batch_size=32):
        self.model.train()  # Set the model to training mode

        # Set the optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = ExponentialLR(optimizer, gamma=self.gamma)

        loop_progress = tqdm(range(self.epochs), leave=True)
        for epoch in loop_progress:
            epoch_loss = 0.0
            for sequences, lengths, targets in train_loader:
                # Zero the gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = self.model(sequences, lengths)

                # Compute the loss
                loss = self.criterion(outputs.squeeze(), targets)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                # Accumulate loss
                epoch_loss += loss.item()
            scheduler.step()

            avg_loss = epoch_loss / len(train_loader)
            # print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')
            loop_progress.set_postfix({f'Epoch [{epoch + 1}/{self.epochs}] Loss': f"{avg_loss:.4f}"})
    def predict(self, test_loader):
        self.model.eval()
        y_pred = []
        # Make predictions
        with torch.no_grad():
            for sequences, lengths, targets in test_loader:
                y_pred.extend(self.model(sequences, lengths))

        return np.asarray(y_pred).flatten()
