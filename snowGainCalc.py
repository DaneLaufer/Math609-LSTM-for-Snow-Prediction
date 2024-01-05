
import pandas as pd
import numpy as np
import pytz


df = pd.read_csv('rawData/altaSnow.csv')


df = df.fillna(method='ffill')
df.index = pd.to_datetime(df.DateTime)

df.index = df.index.tz_localize("US/Mountain").tz_convert(pytz.utc)


snowGained = []
prevInt = 0
for index, row in df.iterrows():
    currentInt = row['Snow Interval']
    if currentInt < prevInt:
        snowGained.append(currentInt)
    else:
        snowGained.append(currentInt-prevInt)
    prevInt = currentInt

df['SnowGain'] = snowGained

df = df['SnowGain']

print(df.head())

df.to_csv('cleanData/SnowGain.csv')


    
