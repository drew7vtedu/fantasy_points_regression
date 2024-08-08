import yaml
import pandas as pd
import os


def load_config(config_path) -> dict:
    """
    loads a file at the given path as a dictionary
    """
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def get_sql_conn_str(config) -> str:
    """
    Return a formatted sql connection str using values from config
    """
    return f"postgresql+psycopg2://{config['sql_username']}:{config['sql_password']}@localhost:{config['sql_port']}/premier_league_data"

def load_cached_data(load_path):
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