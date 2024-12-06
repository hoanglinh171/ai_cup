import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from catboost import CatBoostRegressor, EShapCalcType, EFeaturesSelectionAlgorithm
from statsmodels.tsa.seasonal import STL

import tqdm

def feature_differencing(time_feature, lag):
    diff_feat = time_feature[:-lag]
    diff_feat = np.append(diff_feat.values[:lag], diff_feat)
    return diff_feat
    
    
def add_features(data, time_col):
    data['hour'] = pd.to_datetime(data[time_col]).dt.hour
    data['minute'] = pd.to_datetime(data[time_col]).dt.minute
    
    return data


def time_decompose(data_col, period, season):
    stl = STL(data_col.fillna(data_col.interpolate('linear')), period=period, seasonal=season)
    res = stl.fit()
    seasonal_component = res.seasonal
    trend_component = res.trend

    df_deseason = data_col - seasonal_component - trend_component
    df_deseason_imputed = df_deseason.fillna(df_deseason.interpolate('linear'))
    df_imputed = df_deseason_imputed + seasonal_component + trend_component

    return df_imputed, df_deseason, seasonal_component + trend_component


def prepare_dataset(data, features):
    # Mark missing data
    data['date'] = pd.to_datetime(data['DateTime']).dt.date
    data['is_missing'] = data['Power(mW)'].isnull()
    data['gap_group'] = (~data['is_missing']).cumsum()
    gap_counts = (
        data.groupby('gap_group')['is_missing']
        .sum()
        .reset_index(name='missing_count')
    )
    gap_counts = gap_counts[gap_counts['missing_count'] > 0]

    # Drop the date that have large gap
    # group_lst = gap_counts[gap_counts['missing_count'] > 60]['gap_group'].values
    # missing_date_lst = data[data['gap_group'].isin(group_lst)]['date'].unique()
    # data = data[~data['date'].isin(missing_date_lst)]

    # Impute use decomposition
    for feat in features:
        data[feat], data[f'{feat}_deseason'], data[f'{feat}_season_trend'] = time_decompose(data[feat], 60*30, 15)

    # # Get the complete day only
    # data['date'] = pd.to_datetime(data['DateTime']).dt.date
    # record_counts = data.groupby('date').size()
    # data = data[data['date'].isin(list(record_counts[record_counts == 60].index))]
    data['power_lag1'] = feature_differencing(data['Power(mW)_log_deseason'], 1)
    # train_set['power_lag60'] = feature_differencing(train_set['Power(mW)'], 60)
    data['power_lag2'] = feature_differencing(data['Power(mW)_log_deseason'], 2)
    data['power_lag3'] = feature_differencing(data['Power(mW)_log_deseason'], 3)
    data['power_lag4'] = feature_differencing(data['Power(mW)_log_deseason'], 4)
    data['power_lag5'] = feature_differencing(data['Power(mW)_log_deseason'], 5)

    # get train data and input for test
    hour_dt = pd.to_datetime(data["DateTime"]).dt.time
    data_late = data[(hour_dt >= pd.to_datetime("07:00").time()) & 
                        (hour_dt < pd.to_datetime("17:00").time())]
    
    data_early = data[(hour_dt >= pd.to_datetime("07:00").time()) & 
                        (hour_dt < pd.to_datetime("09:00").time())]
    
    data_to_predict = data_late[data_late['to_predict'] == 1]
    # data_to_predict = data_late
    print(data_to_predict.shape)
    # data_to_predict = data_late

    # drop date with missing value 
    record_counts = data_late.groupby('date')['is_missing'].sum()
    data_late = data_late[data_late['date'].isin(list(record_counts[record_counts == 0].index))]

    record_counts = data_early.groupby('date')['is_missing'].sum()
    data_early = data_early[data_early['date'].isin(list(record_counts[record_counts == 0].index))]
    
    return data_late, data_early, data_to_predict


def scale_min_max(fit_data):
    """
    train_set, test_set: numpy array
    """
    minmax_scaler = MinMaxScaler().fit(fit_data)
    data_scaled = minmax_scaler.transform(fit_data)

    return data_scaled, minmax_scaler


def train_model(train_set):
    train_set = add_features(train_set, 'DateTime')
    features = ['hour', 'minute', 'WindSpeed(m/s)_deseason', 'Pressure(hpa)_deseason',
                'Temperature(°C)_deseason',
                'Humidity(%)_deseason','Sunlight(Lux)_deseason', 
                'power_lag1', 
                'power_lag2', 'power_lag3', 'power_lag4', 'power_lag5',
                'Power(mW)_log_deseason']
    X_features = features[:-1]
    
    train_imputed = train_set[features]
    train_scaled, minmax_scaler = scale_min_max(train_imputed[X_features])

    ## CatBoost

    y_train = train_imputed[['Power(mW)_log_deseason']]
    y_train_scaled, y_scaler = scale_min_max(y_train)
    cat_params = dict(iterations=7000,
                      learning_rate=0.01,
                      depth=11,
                      l2_leaf_reg=30,
                      bootstrap_type='Bernoulli',
                      subsample=0.66,
                      loss_function='MAE',
                      eval_metric = 'MAE',
                      metric_period=100,
                      od_type='Iter',
                      od_wait=30,
                      task_type='GPU',
                      allow_writing_files=False,
                      )

    # Train a LightGBM model for the current fold
    cat_model = CatBoostRegressor(**cat_params)
    cat_model.fit(
        train_scaled,
        y_train_scaled,
        early_stopping_rounds=100
    )
    
    return minmax_scaler, cat_model, y_scaler


def predict_power(test_set, to_predict_data, cat_model, minmax_scaler):
    # date_lst = to_predict_data['date'].unique()
    test_set = add_features(test_set, 'DateTime')
    features = ['hour', 'minute', 'WindSpeed(m/s)_deseason', 'Pressure(hpa)_deseason',
                'Temperature(°C)_deseason',
                'Humidity(%)_deseason','Sunlight(Lux)_deseason', 
                'power_lag1', 
                'power_lag2', 'power_lag3', 'power_lag4', 'power_lag5', 
                'Power(mW)_log_deseason']
    # features = ['Pressure(hpa)','Temperature(°C)','Humidity(%)','Sunlight(Lux)', 'Power(mW)']
    X_features = features[:-1]
    
    # get log
#     test_set[features[1:]] = np.log(test_set[features[1:]] + 1)
    
    test_imputed = test_set[features]
    test_scaled = minmax_scaler.transform(test_imputed[X_features])
    # X_test_pred = lstm_predict(lstm_regressor, forecast_num, look_back_num, test_scaled, to_predict_data)
    X_test_pred = test_scaled

    y_pred = cat_model.predict(X_test_pred).reshape(to_predict_data.shape[0])
    # y_pred = np.exp(y_pred) - 1
#     y_test = np.exp(y_test) - 1

    return X_test_pred, y_pred


if __name__ == '__main__':
    features = ['WindSpeed(m/s)', 'Pressure(hpa)', 'Temperature(°C)', 'Humidity(%)', 'Sunlight(Lux)', 'Power(mW)_log']
    location_lst = range(1, 18, 1)
    look_back_num = 12
    forecast_num = 48
    data_dir = os.getcwd()

    all_predictions = pd.DataFrame(columns = ["serial", "y_pred"])
    for loc in location_lst:
        print(loc, "----------------------------------------------------------------------------------------------------")
        data = pd.read_csv(data_dir + f'/data/add_lag_diff/loc_{loc}.csv')
        data_late, data_early, data_to_predict = prepare_dataset(data, features)

        # data_late_train, data_late_test = data_late.reset_index().loc[:5500, ], data_late.reset_index().loc[5500:, ]
        minmax_scaler, cat_model, y_scaler = train_model(data_late)
        X_test_pred, y_pred = predict_power(data_to_predict, data_to_predict,  
                                            cat_model=cat_model, minmax_scaler=minmax_scaler)
        y_pred_norm = y_scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(-1)
        y_pred_norm = y_pred_norm + data_to_predict["Power(mW)_log_season_trend"]
        y_pred_norm = np.exp(y_pred_norm) - 1

        df = pd.DataFrame(columns = ["serial", "y_pred"])
        df['serial'] = data_to_predict['Serial']
        # df['y_pred'] = (y_pred_norm + data_to_predict['Power(mW)_log_season_comp']).values
        df['y_pred'] = y_pred_norm

        all_predictions = pd.concat([all_predictions, df])

    output = pd.read_csv("upload(no answer).csv")
    output = output.merge(all_predictions, left_on='序號', right_on='serial')
    output['答案'] = output['y_pred']
    output[['序號', '答案']].to_csv('output_stl_deseason.csv', index=False)