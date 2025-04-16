import pandas as pd
import numpy as np
from type_utils import Union, Any

def insert_df_into_df(inserted_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    '''
    inserted_df: pd.DataFrame or pd.Series
        Must have the same columns as df and compatible index types. e.g.
        line = DataFrame({"onset": 30.0, "length": 1.3}, index=[2.5])
    df: pd.DataFrame or pd.Series
        Host df or series
    
    return: pd.DataFrame or pd.Series
        Now the rows of inserted_df are incorporated into df with the indices sorted
        
    Purpose: Insert the rows of one df into another and then sort the rows by index

    Notes: It will break ties in index value by putting the inserted values after the
        original ones when they were in their original position. e.g.
        df = pd.DataFrame({"onset": [30.0,20.0], "length": [1.3,1.2]}, index=range(2))
        df_inserted = pd.DataFrame({"onset": [40.0,50.0], "length": [1.4,1.5]}, index=[0,1])
        print(pandas_utils.insert_df_into_df(df_inserted, df))
        ->     length  onset
           0     1.3   30.0
           1     1.4   40.0
           2     1.2   20.0
           3     1.5   50.0
    '''
    df = df.append(inserted_df, ignore_index=False)
    df = df.sort_index().reset_index(drop=True)
    return df


def get_value_from_row(df: pd.DataFrame, col_name: str, row_value: Any, value_col_name: Union[str, int]) -> Any:
    return df[df[col_name] == row_value][value_col_name].values[0]