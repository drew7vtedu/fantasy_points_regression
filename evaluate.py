import pdb

import pandas as pd
import numpy as np
import argparse
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from preprocess import Preprocessor
from models import BaselineModel, BaselineModel2, NeuralNetRegressor


class Evaluator:

    def __init__(self, args):
        """
        Create a new Evaluator
        """
        self.args = args
        self.models_map = {
            'baseline':
                (BaselineModel,
                 {'group_cols': ['fbref_first_name', 'fbref_last_name'], 'col_to_shift': 'total_points'}),
            'baseline2': (BaselineModel2,
                 {'group_cols': ['fbref_first_name', 'fbref_last_name'], 'col_to_shift': 'total_points'}),
            'random_forest': (RandomForestRegressor, {}),
            'gradient_boost': (GradientBoostingRegressor, {}),
            'linear': (LinearRegression, {}),
            'nn': (NeuralNetRegressor, {'layers': [40] * 2, 'activation_function': 'leaky_relu', 'lr': 0.020444883080411782, 'epochs': 45, 'gamma': 0.8723319384111253}
)
        }
        self.models = [self.models_map[model] for model in self.args.models_to_evaluate]

        self.preprocessor = Preprocessor(args=args)

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

        parser.add_argument('--models_to_evaluate', type=list, required=False,
                            default=['baseline', 'random_forest', 'gradient_boost', 'linear', 'nn'])

        return parser

    def main(self) -> None:
        """
        Run evaluation on all models
        :return: None
        """
        results = {}
        data = self.preprocessor.main()
        drop_cols = ['season', 'team', 'fbref_first_name', 'fbref_last_name', 'total_points']
        X = data[[x for x in data.columns if x != 'target']]
        y = data['target']

        X_train = X.loc[X.season != '22/23'].drop(drop_cols, axis=1)
        X_test = X.loc[X.season == '22/23'].drop(drop_cols, axis=1)

        y_train = y.iloc[X_train.index]
        y_test = y.iloc[X_test.index]

        # baseline_index = data.loc[data.season == '22/23'].index

        for model_string in self.args.models_to_evaluate:
            model_func, model_args = self.models_map[model_string]
            model = model_func(**model_args)
            if 'baseline' in model_string:
                res = model.predict(data)
                results_index = np.where(np.isnan(res) == False)[0]
                # results[model_string] = mean_squared_error(y.iloc[baseline_index].values[results_index], res[results_index])
                results[model_string] = mean_absolute_error(y.values[results_index],
                                                            res[results_index])
            else:

                if hasattr(model, 'fit'):
                    model = model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                results[model_string] = mean_absolute_error(y_test, y_pred)

        print('=' * 25)
        for model_string in results:
            print(f"{model_string}:")
            print(f"\tMSE: {results[model_string]}")


if __name__ == '__main__':
    args = Evaluator.init_command_line_args().parse_args()
    evaluator = Evaluator(args)
    evaluator.main()
