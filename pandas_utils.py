import pandas as pd
import numpy as np


def insert_df_into_df(inserted_df, df):
    '''
    inserted_df: pd.DataFrame
        Must have the same columns as df and compatible index types. e.g.
        line = DataFrame({"onset": 30.0, "length": 1.3}, index=[2.5])
    df: pd.DataFrame
        Host df
    
    return: pd.DataFrame
        Now the rows of inserted_df are incorporated into df with the indices sorted
        
    Purpose: Insert the rows of one df into another and then sort the rows by index
    '''
    df = df.append(inserted_df, ignore_index=False)
    df = df.sort_index().reset_index(drop=True)
    return df
