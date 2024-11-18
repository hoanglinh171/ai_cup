import numpy as np
import pandas as pd
import os
from sklearn.model_selection import TimeSeriesSplit

data_dir = os.getcwd()

def avg_data(i):
    print(i)
    df1 = pd.read_csv(data_dir + "/data/36_TrainingData/L{i}_Train.csv".format(i=i),
                    index_col="DateTime", parse_dates=["DateTime"])
    df2 = pd.read_csv(data_dir + "/data/36_TrainingData/L{i}_Train_2.csv".format(i=i),
                    index_col="DateTime", parse_dates=["DateTime"])
    df = pd.concat([df1, df2])

    # Replace unrealistic pressure with central moving average q = 6
    df["Pressure(hpa)"] = np.where(df["Pressure(hpa)"] >= 1100, np.nan, df["Pressure(hpa)"])
    df["Pressure(hpa)"] = df["Pressure(hpa)"].fillna(df["Pressure(hpa)"].rolling(6, center=True).mean())

    # Replace unrealistic humidity with central moving average q = 6
    # df["Humidity(%)"] = np.where(df["Humidity(%)"] > 100, np.nan, df["Humidity(%)"])
    # df["Humidity(%)"] = df["Humidity(%)"].fillna(df["Humidity(%)"].rolling(6, center=True).mean())

    # 10 minute interval
    start = (df.index.floor("min") - pd.to_timedelta(df.index.minute % 10, "minute"))[0]
    df_avg = df.resample("10min", origin=start).mean()
    df_avg['LocationCode'] = df_avg['LocationCode'].ffill()
    df_avg['Serial'] = df_avg.index.strftime('%Y%m%d%H%M')
    df_avg['Serial'] = df_avg['Serial'] + df_avg['LocationCode'].astype(int).astype(str).str.zfill(2)
    second_col = df_avg.pop('Serial')
    df_avg.insert(1, 'Serial', second_col)


    # Filter 7am to 5pm
    hour_dt = df_avg.index.time
    filtered_df = df_avg[(hour_dt >= pd.to_datetime("07:00").time()) & 
                        (hour_dt < pd.to_datetime("17:00").time())]

    filtered_df.to_csv(data_dir + "/data/avg_10min/loc_{i}.csv".format(i=i))
    # 5 folds split by days
    # date_dt = filtered_df.index.date
    # unique_data = pd.Series(date_dt).unique()
    #
    # tscv = TimeSeriesSplit(n_splits=5)
    # for j, (train_index, test_index) in enumerate(tscv.split(unique_data)):
    #     train_days = unique_data[train_index]
    #     test_days = unique_data[test_index]
    #
    #     train = filtered_df[np.isin(date_dt, train_days)]
    #     test = filtered_df[np.isin(date_dt, test_days)]
    #
    #     os.makedirs(data_dir + "/avg_data_10min_wh/loc_{i}/fold_{j}".format(j=j, i=i), exist_ok=True)
    #     train.to_csv(data_dir + "/avg_data_10min_wh/loc_{i}/fold_{j}/train.csv".format(j=j, i=i))
    #     test.to_csv(data_dir + "/avg_data_10min_wh/loc_{i}/fold_{j}/test.csv".format(j=j, i=i))
    #
    #     os.makedirs(data_dir + "/avg_data_10min_wh/all_location/fold_{j}".format(j=j, i=i), exist_ok=True)
    #     train_path = "/avg_data_10min_wh/all_location/fold_{j}/train.csv".format(j=j)
    #     test_path = "/avg_data_10min_wh/all_location/fold_{j}/test.csv".format(j=j)
    #     train.to_csv(data_dir + train_path, mode='a', header=False)
    #     test.to_csv(data_dir + test_path, mode='a', header=False)


def differencing(series, lag):
    series_diff = series[lag:] - series[:-lag]
    series_diff = np.append(series_diff[:lag], series_diff)

    return series_diff

def lagging(series, lag):
    series_lag = series[lag:]
    series_lag = np.append(series_lag[:lag], series_lag)

    return series_lag

def add_features(features_df):
    features_lst = features_df.columns

    feat_name = []
    added_feat = []
    for feat in features_lst:
        # Normal scale
        feature_data = features_df[feat].values
        lag_1 = lagging(feature_data, 1)
        added_feat.append(lag_1)
        feat_name.append(feat + "_lag1")

        lag_60 = lagging(feature_data, 60)
        added_feat.append(lag_60)
        feat_name.append(feat + "_lag60")

        diff_1 = differencing(feature_data, 1)
        added_feat.append(diff_1)
        feat_name.append(feat + "_diff1")

        diff_60 = differencing(feature_data, 60)
        added_feat.append(diff_60)
        feat_name.append(feat + "_diff60")

        diff_1_60 = differencing(diff_1, 60)
        added_feat.append(diff_1_60)
        feat_name.append(feat + "_diff_1_60")

        # Log scale
        feature_data = np.log(features_df[feat].values + 1)
        added_feat.append(feature_data)
        feat_name.append(feat + "_log")

        lag_1 = lagging(feature_data, 1)
        added_feat.append(lag_1)
        feat_name.append(feat + "_lag1_log")

        lag_60 = lagging(feature_data, 60)
        added_feat.append(lag_60)
        feat_name.append(feat + "_lag60_log")

        diff_1 = differencing(feature_data, 1)
        added_feat.append(diff_1)
        feat_name.append(feat + "_diff1_log")

        diff_60 = differencing(feature_data, 60)
        added_feat.append(diff_60)
        feat_name.append(feat + "_diff60_log")

        diff_1_60 = differencing(diff_1, 60)
        added_feat.append(diff_1_60)
        feat_name.append(feat + "_diff_1_60_log")

    added_feat = np.transpose(np.array(added_feat))
    added_features_df = pd.DataFrame(added_feat, columns=feat_name)

    return added_features_df


if __name__ == "__main__":
    to_predict = pd.read_csv(data_dir + "/data/to_predict.csv")
    for i in range(1, 18):
        df = pd.read_csv(data_dir + "/data/avg_10min/loc_{i}.csv".format(i=i), index_col="DateTime", parse_dates=["DateTime"])
        features = df.columns[2:]
        added_features = add_features(df[features])
        added_features.index = df.index

        new_df = pd.concat([df, added_features], axis=1)
        new_df['to_predict'] = np.where(new_df['Serial'].astype(str).isin(to_predict.astype(str).values.reshape(1, -1)[0]), 1, 0)
        third_col = new_df.pop('to_predict')
        new_df.insert(2, 'to_predict', third_col)

        new_df.to_csv(data_dir + "/data/add_lag_diff/loc_{i}.csv".format(i=i))

        