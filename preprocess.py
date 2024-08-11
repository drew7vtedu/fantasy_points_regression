import pdb
import pandas as pd
import numpy as np
import os
from sqlalchemy import create_engine
import argparse
from sklearn.preprocessing import OneHotEncoder

import util_funcs as util

class Preprocessor:

    def __init__(self, args):
        """
        Create a new Preprocessor
        :param args: command line arguments to determine behavior
        """
        self.args = args
        self.add_bps_features = True
        with open(self.args.query_path, 'r') as query_file:
            self.fbref_data_query = query_file.read()
        self.config = util.load_config(self.args.config_path)
        self.sql_conn_str = f"postgresql+psycopg2://{self.config['sql_username']}:{self.config['sql_password']}@localhost:{self.config['sql_port']}/premier_league_data"
        self.duplicate_column_names = ['assists', 'yellow_cards', 'red_cards', 'clean_sheets']
        self.necessary_fpl_features = ['first_name', 'second_name', 'season', 'total_points']
        self.extra_fpl_features = self.args.extra_fpl_features
        self.drop_after_matching = ['fpl_first_name', 'fpl_second_name', 'first_name', 'second_name']
        self.group_cols = ['fbref_first_name', 'fbref_last_name']
        self.col_to_shift = 'total_points'

    @staticmethod
    def init_command_line_args():
        """
        Create command line arguments for base class and return parser
        """
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        parser.add_argument('--config_path', type=str, required=False, default='_config_.yaml',
                            help='Path to the config file to load.')
        parser.add_argument('--fpl_data_path', type=str, required=False, default='data/fpl',
                            help='Path to the directory of fpl player data.')
        parser.add_argument('--query_path', type=str, required=False, default='queries/all_source_data.sql',
                            help='Path to the source data query for fbref player data.')
        parser.add_argument('--extra_fpl_features', type=list, required=False, default=[],
                            help='any extra fpl features you wish to add to enrich data')

        return parser

    def load_cached_data(self, load_path) -> pd.DataFrame:
        """
        Load data which has been saved in csvs.
        """
        fpl_df = pd.DataFrame()
        dir = os.listdir(load_path)
        csvs = [x for x in dir if '.csv' in x]
        for season in csvs:
            season_str = season[-9:-4]
            df = pd.read_csv(load_path+'/'+season)
            df['season'] = season_str.replace('_', '/')
            fpl_df = pd.concat([fpl_df, df])

        return fpl_df

    def load_fbref_data(self) -> pd.DataFrame:
        """
        Load data from the database into a dataframe
        :return: DataFrame containing fbref data
        """
        engine = create_engine(self.sql_conn_str)
        with engine.connect() as conn:
            fbref_data = pd.read_sql(self.fbref_data_query, conn)
        return fbref_data

    def preprocess_fpl_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Do some preprocessing steps before joining the data
        :param df: contains fpl player data
        :return: the cleaned fpl data
        """
        features = self.necessary_fpl_features
        features.extend(self.extra_fpl_features)
        fpl_data = df[features]
        fpl_data = fpl_data.drop_duplicates(subset=['season', 'first_name', 'second_name'])

        return fpl_data

    def preprocess_fbref_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Do some preprocessing steps before joining the data
        :param df: contains fbref player data
        :return: the cleaned fbref data
        """
        fbref_data = df.copy()
        fbref_data['pos'] = fbref_data['pos'].str[:2]
        encoder = OneHotEncoder(sparse_output=False)
        positions_encoded = encoder.fit_transform(fbref_data[['pos']])
        positions_encoded = pd.DataFrame(positions_encoded, columns=encoder.get_feature_names_out())
        fbref_data = pd.concat([fbref_data, positions_encoded], axis=1)
        fbref_data = fbref_data.drop(columns='pos')

        fbref_data['age'] = fbref_data['age'].astype(int)
        fbref_data['minutes_played'] = fbref_data['minutes_played'].astype(float)

        fbref_data = fbref_data.drop_duplicates(subset=['fbref_first_name', 'fbref_last_name', 'season'], keep='first')

        return fbref_data

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
        df_sorted['shifted_' + self.col_to_shift] = df_sorted.groupby(self.group_cols)[self.col_to_shift].shift(-1)

        # Restore the original order of the DataFrame
        df_sorted = df_sorted.sort_index()

        return df_sorted

    def main(self, use_fpl: bool=True) -> pd.DataFrame:
        """
        main method which loads and preprocesses the data
        :return: DataFrame containing data ready for modeling
        """
        if use_fpl:
            fpl_data = self.load_cached_data(self.args.fpl_data_path)
            fpl_data = self.preprocess_fpl_data(fpl_data)
        fbref_data = self.load_fbref_data()
        fbref_data = self.preprocess_fbref_data(fbref_data)
        if use_fpl:
            join_df = fbref_data.merge(fpl_data,
                                       left_on=['fpl_first_name', 'fpl_second_name', 'season'],
                                       right_on=['first_name', 'second_name', 'season'])
            join_df = join_df.drop(columns=self.drop_after_matching)
            if self.add_bps_features:
                og_query = self.fbref_data_query
                with open('queries/bonus_point_query.sql', 'r') as query_file:
                    self.fbref_data_query = query_file.read()
                bps_data = self.load_fbref_data()
                bps_data = bps_data.dropna()
                self.fbref_data_query = og_query
                join_df = join_df.merge(bps_data, left_on=['fbref_first_name', 'fbref_last_name', 'season', 'team'],
                                           right_on=['first_name', 'last_name', 'season', 'team'])
                join_df = join_df.drop(columns=['first_name', 'last_name'], axis=1)
                join_df['dribbles'] = join_df['dribbles'].astype(int)
                join_df['tackled'] = join_df['tackled'].astype(int)
            join_df = self.add_shifted_column(join_df)
            join_df = join_df.rename({f"shifted_{self.col_to_shift}": 'target'}, axis=1)
            join_df = join_df.loc[join_df.season != '23/24']
            join_df = join_df.dropna().reset_index(drop=True)
            return join_df
        else:
            return fbref_data


if __name__ == '__main__':
    args = Preprocessor.init_command_line_args().parse_args()
    preprocessor = Preprocessor(args)
    preprocessor.main()
