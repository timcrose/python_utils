import pandas_utils
import pandas as pd

df = pd.Series([30.0,20.0], index=range(2))
df_inserted = pd.Series([40.0,50.0], index=[0,1])
print(df)
print(pandas_utils.insert_df_into_df(df_inserted, df))
