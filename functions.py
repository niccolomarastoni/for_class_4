import pandas as pd
import streamlit as st

def get_data(url : str) -> pd.DataFrame:
    '''
    get_data: given a url, outputs a dataframe with the right data
    attributes:
    url: string

    examples:
    >> url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/refs/heads/master/titanic.csv'
    >> titanic_df = get_data(url)
    '''
    try:
        data_df = pd.read_csv(url)
        return data_df
    except:
        print('Not a proper csv file')

def data_exploration(titanic_df):
    titanic_df.head(2)
    titanic_df.info()
    titanic_df.describe().T
    titanic_df.Sex.unique()