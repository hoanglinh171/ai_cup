import numpy as np
import pandas as pd
import os
from sklearn.model_selection import TimeSeriesSplit

data_dir = os.getcwd()

for i in range(1, 18):
    df = pd.read_csv(data_dir + "/36_TrainingData/L{i}_Train.csv".format(i=i), 
                     index_col="DateTime", parse_dates=["DateTime"])
    # Replace unrealistic pressure with central moving average q = 6
    df["Pressure(hpa)"] = np.where(df["Pressure(hpa)"] >= 1100, np.nan, df["Pressure(hpa)"])
    df["Pressure(hpa)"] = df["Pressure(hpa)"].fillna(df["Pressure(hpa)"].rolling(6, center=True).mean())

    # Replace unrealistic humidity with central moving average q = 6
    df["Humidity(%)"] = np.where(df["Humidity(%)"] > 100, np.nan, df["Humidity(%)"])
    df["Humidity(%)"] = df["Humidity(%)"].fillna(df["Humidity(%)"].rolling(6, center=True).mean())

    # 10 minute interval
    start = (df.index.floor("min") - pd.to_timedelta(df.index.minute % 10, "minute"))[0]
    df_avg = df.resample("10min", origin=start).mean()
    df_avg['LocationCode'] = df_avg['LocationCode'].ffill()

    # 5 folds split
    tscv = TimeSeriesSplit(n_splits=5)
    for j, (train_index, test_index) in enumerate(tscv.split(df_avg)):
        train = df_avg[df_avg.reset_index().index.isin(train_index)]
        test = df_avg[df_avg.reset_index().index.isin(train_index)]

        # os.makedirs(data_dir + "/avg_data_10min/loc_{i}/fold_{j}".format(j=j, i=i))
        train.to_csv(data_dir + "/avg_data_10min/loc_{i}/fold_{j}/train.csv".format(j=j, i=i))
        train.to_csv(data_dir + "/avg_data_10min/loc_{i}/fold_{j}/test.csv".format(j=j, i=i))