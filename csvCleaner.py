import pandas as pd
import numpy as np
import pytz

# df = pd.read_csv('PC064.csv')
# df.index = pd.to_datetime(df.Date_Time)
# df= df[['air_temp_set_1', 'relative_humidity_set_1', 'wind_speed_set_1', 'wind_direction_set_1', 'wind_gust_set_1']]
# df = df.rename(columns= {"air_temp_set_1": 'PC064Temp',
# 'relative_humidity_set_1': 'PC064Humidity',
# 'wind_speed_set_1': 'PC064WindSpeed',
# 'wind_direction_set_1': 'PC064WindDir',
# 'wind_gust_set_1': 'PC064WindGust'})
# df = df.resample('1H').agg({'PC064Temp': np.mean, 'PC064Humidity': np.mean, 'PC064WindSpeed': np.mean, 'PC064WindDir': np.median, 'PC064WindGust': np.max})

# df = df.fillna(method='ffill')
# print(df.isna().sum())
# df.to_csv('PC064Cleaned.csv')

# df = pd.read_csv('F8379.csv')

# df.index = pd.to_datetime(df.Date_Time)
# df = df[df.columns[2:]]
# df = df.resample('1H').mean()
# # df = df.fillna({'KPVUCloud3': 1.0, 'KPVUCloud1': 1.0, 'KPVUCloud2': 1.0, 'KPVUVis': 10.0})
# df = df.fillna(method='ffill')
# print(df.head())
# print(df.isna().sum())

# df.to_csv('cleanData/F8379Cleaned.csv')


df = pd.read_csv('rawData/AGD (2).csv')

times = []
for index, row in df.iterrows():
    trimmedDate = row['Date_Time'][:-3]
    tz = pytz.timezone("US/Mountain")
    time = pd.to_datetime(trimmedDate).tz_localize(tz).tz_convert(pytz.utc)
    times.append(time)

df['Date_Time'] = times
df.index = pd.to_datetime(df.Date_Time)
snowGained = []
prevInt = 0
for index, row in df.iterrows():
    currentInt = row['AGDSnowInt']
    if currentInt < prevInt:
        if currentInt > prevInt -.3:
            snowGained.append(0)
            continue
        else:
            if currentInt > 1:
                snowGained.append(0)
            else:
                snowGained.append(currentInt)


    else:
        snowGained.append(currentInt-prevInt)
    prevInt = currentInt

df['SnowGain'] = snowGained
df = df[['SnowGain' , 'AGDTemp', 'AGDHumid' ,'AGDWindSpeed','AGDWindDir','AGDWindGust']]

df = df.resample('1H').agg({'SnowGain': np.sum, 'AGDTemp': np.mean, 'AGDHumid': np.mean, 'AGDWindSpeed': np.mean, 'AGDWindDir': np.median, 'AGDWindGust': np.max})
df = df.fillna(method='ffill')

print(df.head())
print(df.isna().sum())


df.to_csv('cleanData/AGDCleaned.csv')


