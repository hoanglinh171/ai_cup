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

    # Filter 7am to 5pm
    hour_dt = df_avg.index.time
    filtered_df = df_avg[(hour_dt >= pd.to_datetime("07:00").time()) & 
                          (hour_dt < pd.to_datetime("17:00").time())]
    
    # 5 folds split by days
    date_dt = filtered_df.index.date
    unique_data = pd.Series(date_dt).unique()

    tscv = TimeSeriesSplit(n_splits=5)
    for j, (train_index, test_index) in enumerate(tscv.split(unique_data)):
        train_days = unique_data[train_index]
        test_days = unique_data[test_index]

        train = filtered_df[np.isin(date_dt, train_days)]
        test = filtered_df[np.isin(date_dt, test_days)]

        os.makedirs(data_dir + "/avg_data_10min_wh/loc_{i}/fold_{j}".format(j=j, i=i), exist_ok=True)
        train.to_csv(data_dir + "/avg_data_10min_wh/loc_{i}/fold_{j}/train.csv".format(j=j, i=i))
        test.to_csv(data_dir + "/avg_data_10min_wh/loc_{i}/fold_{j}/test.csv".format(j=j, i=i))

        os.makedirs(data_dir + "/avg_data_10min_wh/all_location/fold_{j}".format(j=j, i=i), exist_ok=True)
        train_path = "/avg_data_10min_wh/all_location/fold_{j}/train.csv".format(j=j)
        test_path = "/avg_data_10min_wh/all_location/fold_{j}/test.csv".format(j=j)
        train.to_csv(data_dir + train_path, mode='a', header=False)
        test.to_csv(data_dir + test_path, mode='a', header=False)