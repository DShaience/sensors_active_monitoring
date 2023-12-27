from collections import OrderedDict
from typing import List

import numpy as np
import pandas as pd

PERF_CAP_REASON = {
    # Taken from:
    #   https://www.techpowerup.com/forums/threads/gpu-z-perfcap-log-number-meanings.202433/
    1: "NV_GPU_PERF_POLICY_ID_SW_POWER",        # Power. Indicating perf is limited by total power limit.
    2: "NV_GPU_PERF_POLICY_ID_SW_THERMAL",      # Thermal. Indicating perf is limited by temperature limit.
    4: "NV_GPU_PERF_POLICY_ID_SW_RELIABILITY",  # Reliability. Indicating perf is limited by reliability voltage.
    8: "NV_GPU_PERF_POLICY_ID_SW_OPERATING",    # Operating. Indicating perf is limited by max operating voltage.
    16: "NV_GPU_PERF_POLICY_ID_SW_UTILIZATION"  # Utilization. Indicating perf is limited by GPU utilization.
}


def extract_reasons(code):
    reasons = []
    for key, value in PERF_CAP_REASON.items():
        key = int(key)
        if code & key:
            reasons.append(value)
    return reasons


def convert_columns_data_types(df: pd.DataFrame, datetime_cols: list, numeric_cols: list):
    for col in df.columns:
        print(col)
        if col in datetime_cols:
            df[col] = pd.to_datetime(df[col])
        elif col in numeric_cols:
            try:
                df[col] = df[col].str.strip().replace('-', np.nan)
            except:
                pass
            df[col] = pd.to_numeric(df[col])
        else:
            pass

    return df


def read_sensors_data_from_file(file: str):
    sensors = pd.read_csv(file, encoding='unicode_escape')

    # removing repeating header by comparing the first column name to all rows' values.
    # If this matches, it means that the row is probably a repeat of the header
    first_column = sensors.columns[0]
    sensors = sensors[sensors[first_column] != first_column]

    # remove empty columns
    sensors.dropna(axis=1, how='all', inplace=True)

    # rename columns to eliminate leading and trailing spaces
    columns = {col: col.strip() for col in sensors.columns}
    sensors.rename(columns=columns, inplace=True)
    sensors_with_recalculated_dtypes = convert_columns_data_types(sensors, datetime_cols=['Date'], numeric_cols=[col for col in sensors.columns if col != 'Date'])

    sensors_with_recalculated_dtypes.sort_values(by=['Date'], ascending=True, inplace=True)
    sensors_with_recalculated_dtypes.reset_index(drop=True, inplace=True)
    return sensors_with_recalculated_dtypes


def calc_time_delta_seconds(series_a: pd.Series, series_b: pd.Series) -> pd.Series:
    return (series_b-series_a).dt.total_seconds()


def add_nan_values_on_time_gap(df: pd.DataFrame, date_column: str, cols_to_set: List[str], gap_threshold_seconds: float = 3600.0):
    df['time_gap_sec'] = calc_time_delta_seconds(df[date_column], df[date_column].shift(-1))
    df.loc[df['time_gap_sec'] >= gap_threshold_seconds, cols_to_set] = np.nan
    return df


def calc_statistics(sensor_data: np.ndarray, sensor_name: str) -> dict:
    features = OrderedDict({"Sensor": sensor_name})
    features['max'] = np.max(sensor_data)
    features['min'] = np.min(sensor_data)
    features['mean'] = np.mean(sensor_data)
    features['median'] = np.median(sensor_data)
    features['quantile_0.1'] = np.quantile(sensor_data, 0.1)
    features['quantile_0.95'] = np.quantile(sensor_data, 0.95)
    features['min'] = np.min(sensor_data)
    return features
