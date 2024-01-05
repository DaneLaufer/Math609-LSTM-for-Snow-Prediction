import pandas as pd
import numpy as np

df = pd.read_csv('cleanData/AGDCleaned.csv')



dfToAdd = pd.read_csv('cleanData/KHCRCleaned.csv')
df = pd.merge(df, dfToAdd, how='outer', on='Date_Time')


dfToAdd = pd.read_csv('cleanData/F8379Cleaned.csv')
df = pd.merge(df, dfToAdd, how='outer', on='Date_Time')


dfToAdd = pd.read_csv('cleanData/KU42Cleaned.csv')
df = pd.merge(df, dfToAdd, how='outer', on='Date_Time')

dfToAdd = pd.read_csv('cleanData/KPVUCleaned.csv')
df = pd.merge(df, dfToAdd, how='outer', on='Date_Time')

dfToAdd = pd.read_csv('cleanData/KSLCCleaned.csv')
df = pd.merge(df, dfToAdd, how='outer', on='Date_Time')

df = df.dropna(how='any',axis=0) 


df.index = pd.to_datetime(df.Date_Time)
df.drop(['Date_Time'], axis=1, inplace=True)

print(df)
print(df.isna().sum())

df.to_csv('fullData.csv')
