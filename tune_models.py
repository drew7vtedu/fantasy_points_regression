import pdb

import numpy as np
import pandas as pd
import torch
import optuna
import argparse
from sklearn.metrics import mean_absolute_error

from models import NeuralNetRegressor
from preprocess import Preprocessor


class ModelTuner:

    def __init__(self, args) -> None:
        """
        make a new ModelTuner
        :param args: command line arguments
        """
        self.args = args
        self.loss_metric = mean_absolute_error
        self.preprocessor = Preprocessor(args)
        self.X_train, self.y_train, self.X_test, self.y_test = self.get_data()

    @staticmethod
    def init_command_line_args() -> argparse.ArgumentParser:
        """
        Create command line arguments for base class and return parser
        """
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        parser.add_argument('--config_path', type=str, required=False, default='_config_.yaml',
                            help='Path to the config file to load.')
        parser.add_argument('--fpl_data_path', type=str, required=False, default='data/fpl',
                            help='Path to the directory of fpl player data.')
        parser.add_argument('--query_path', type=str, required=False, default='queries/source_data.sql',
                            help='Path to the source data query for fbref player data.')
        parser.add_argument('--extra_fpl_features', type=list, required=False, default=[],
                            help='any extra fpl features you wish to add to enrich data')
        parser.add_argument('--n_trials', type=int, required=False, default=100,
                            help='number of trials for optuna to optimize.')

        return parser

    def get_data(self) -> tuple:
        """
        Get data and split it into training and test sets
        :return: tuple containing (X_train, y_train, X_test, y_test)
        """
        data = self.preprocessor.main()
        drop_cols = ['season', 'team', 'fbref_first_name', 'fbref_last_name', 'total_points']
        X = data[[x for x in data.columns if x != 'target']]
        y = data['target']

        X_train = X.loc[X.season != '21/22'].drop(drop_cols, axis=1)
        X_test = X.loc[X.season == '21/22'].drop(drop_cols, axis=1)

        y_train = y.iloc[X_train.index]
        y_test = y.iloc[X_test.index]

        return X_train, y_train, X_test, y_test

    def objective(self, trial):
        n_layers = trial.suggest_int('n_layers', 1, 10)
        nodes_per_layer = trial.suggest_int('nodes_per_layer', 5, 100)
        params = {
            'activation_function': trial.suggest_categorical('activation_function',
                                                             ['tanh', 'relu', 'sigmoid', 'leaky_relu']),
            'lr': trial.suggest_float('lr', 0.000001, 0.1),
            'layers': [nodes_per_layer] * n_layers,
            'epochs': trial.suggest_int('epochs', 1, 100),
            'gamma': trial.suggest_float('gamma', 0.1, 0.99)
        }

        model = NeuralNetRegressor(**params)
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)

        return mean_absolute_error(self.y_test, y_pred)

    def main(self):
        study = optuna.create_study()
        study.optimize(self.objective, n_trials=self.args.n_trials)

        best_params = study.best_params
        print(best_params)


if __name__ == '__main__':
    args = ModelTuner.init_command_line_args().parse_args()
    tuner = ModelTuner(args)
    tuner.main()
