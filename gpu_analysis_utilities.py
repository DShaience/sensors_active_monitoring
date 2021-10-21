from typing import List
import numpy as np
import pandas as pd
import glob
import os
from matplotlib import pyplot as plt
import seaborn as sns
from collections import OrderedDict
from visualization_utils import correlation_matrix
sns.set(color_codes=True)


def parse_line_to_array(line: str, delimiter: str = ',') -> List[str]:
    line_arr = line.split(delimiter)
    return [col.strip() for col in line_arr if col.strip() != '']


def parse_line_array_to_dict(keys, array):
    assert len(keys) == len(array)
    zip_iterator = zip(keys, array)
    return dict(zip_iterator)


def get_file_header(file_path: str) -> str:
    with open(file_path) as f:
        first_line = f.readline().strip()
    f.close()
    return first_line


def read_parse_sensors_data_to_dataframe(file_path):
    # global sensors
    header = get_file_header(file_path)
    parsed_header = parse_line_to_array(header)
    list_of_dict_values = []
    with open(file_path) as f:
        for index, line in enumerate(f):
            if (parsed_header[0] in line.strip()) or (parsed_header[1] in line.strip()):
                continue
            line_as_arr = parse_line_to_array(line)
            list_of_dict_values.append(parse_line_array_to_dict(parsed_header, line_as_arr))
    f.close()
    sensors = pd.DataFrame(list_of_dict_values)
    return sensors


def re_assign_data_types(df: pd.DataFrame, datetime_cols: list, numeric_cols: list):
    for col in df.columns:
        if col in datetime_cols:
            df[col] = pd.to_datetime(df[col])
        elif col in numeric_cols:
            df[col] = df[col].replace('-', np.nan)
            df[col] = pd.to_numeric(df[col])
        else:
            pass

    return df


def get_sensors_files_from_path(path: str):
    files = list(glob.glob(path + os.sep + '*.txt', recursive=False))
    return files


def read_and_concat_sensors_data_from_files(files: List[str]):
    sensors_list_dfs = []
    for file in files:
        sensors = read_parse_sensors_data_to_dataframe(file)
        sensors_with_recalculated_dtypes = re_assign_data_types(sensors, datetime_cols=['Date'], numeric_cols=[col for col in sensors.columns if col != 'Date'])
        sensors_list_dfs.append(sensors_with_recalculated_dtypes)
    sensors_all = pd.concat(sensors_list_dfs)
    sensors_all.sort_values(by=['Date'], ascending=True, inplace=True)
    sensors_all.reset_index(drop=True, inplace=True)
    return sensors_all


def calc_time_delta_seconds(series_a: pd.Series, series_b: pd.Series) -> pd.Series:
    return (series_b-series_a).dt.total_seconds()


def add_nan_values_on_time_gap(df: pd.DataFrame, date_column: str, cols_to_set: List[str], gap_threshold_seconds: float = 3600.0):
    df['time_gap_sec'] = calc_time_delta_seconds(df[date_column], df[date_column].shift(-1))
    df.loc[df['time_gap_sec'] >= gap_threshold_seconds, cols_to_set] = np.nan
    return df


def calc_statistics(sensor_data: np.ndarray, sensor_name: str) -> dict:
    features = OrderedDict({"Sensor": sensor_name})
    # sensor_data = sensor.dropna()
    features['max'] = np.max(sensor_data)
    features['min'] = np.min(sensor_data)
    features['mean'] = np.mean(sensor_data)
    features['median'] = np.median(sensor_data)
    features['quantile_0.1'] = np.quantile(sensor_data, 0.1)
    features['quantile_0.95'] = np.quantile(sensor_data, 0.95)
    features['min'] = np.min(sensor_data)
    return features


if __name__ == '__main__':
    path = r'C:\Users\Shay\Desktop\Desktop items\overclocking\Sensors_data'
    files = get_sensors_files_from_path(path)
    # files = [file for file in files if "RTX 3070 Ti Asus 04 CPU-OC_01.txt" not in file]
    # files = [file for file in files if "RTX 3070 Ti Asus 05 intel core i7-7700k CPU Game - Horizon Zero Dawn.txt" in file]
    # files = [file for file in files if "RTX 3070 Ti Asus 05 intel core i7-7700k CPU Game - Control.txt" in file]
    files = [file for file in files if "RTX 3070 Ti Asus 05 intel core i7-7700k CPU Game - Shadow of the Tomb Raider.txt" in file]
    sensors_all = read_and_concat_sensors_data_from_files(files)

    cols_to_set = [col for col in sensors_all if col != 'Date']
    sensors_nan_gaps = add_nan_values_on_time_gap(sensors_all, 'Date', cols_to_set)

    target_columns = ['CPU Temperature [°C]', 'GPU Temperature [°C]', 'Memory Temperature [°C]', 'Hot Spot [°C]',
                      'GPU Chip Power Draw [W]', 'System Memory Used [MB]', 'Fan 1 Speed (%) [%]', 'Fan 2 Speed (%) [%]']

    n_rows = 4
    n_cols = max(1, int(round(len(target_columns) / n_rows + 0.5, 0)))
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols)

    features_list = []
    for i in range(len(target_columns)):
        target_col = target_columns[i]
        ax = axes.flatten()[i]
        data = sensors_nan_gaps.loc[~sensors_nan_gaps[target_col].isna(), target_col].values
        ax.plot(range(len(data)), data)
        ax.title.set_text(target_col)
        features_list.append(calc_statistics(data, target_col.replace(' ', '_')))
    if n_cols * n_rows > len(target_columns):
        for ax in axes.flatten()[len(target_columns):]:
            ax.remove()

    sensors_features = pd.DataFrame(features_list)
    print(sensors_features.to_string())

    fig.tight_layout(h_pad=-1)
    plt.show()
    print(n_rows)

    # target_col = 'GPU Temperature [°C]'
    # target_col = 'Power Consumption (%) [% TDP]'
    # target_col = 'CPU Temperature [°C]'
    # target_col = 'Memory Temperature [°C]'
    # target_col = 'System Memory Used [MB]'
    # target_col = 'GPU Load [%]'
    # target_col = 'Board Power Draw [W]'
    # target_col = 'Hot Spot [°C]'
    # target_col = 'Fan 1 Speed (RPM) [RPM]'
    # target_col = 'Fan 2 Speed (RPM) [RPM]'
    # target_col = 'Fan 1 Speed (%) [%]'
    # target_col = 'Fan 2 Speed (%) [%]'
    # target_col = 'GPU Chip Power Draw [W]'


# old-cpu
# E:\development\.virtual_env\sensors_active_monitoring\Scripts\python.exe E:/development/sensors_active_monitoring/gpu_analysis_utilities.py
# Memory Temperature [°C] max: 74.0
# Memory Temperature [°C] min: 34.0
# Memory Temperature [°C] mean: 55.21469448033967
# Memory Temperature [°C] median: 56.0
# Memory Temperature [°C] quantile 0.1: 42.0
# Memory Temperature [°C] quantile 0.95: 68.0
#
# Process finished with exit code 0

# i7-7700k
# E:\development\.virtual_env\sensors_active_monitoring\Scripts\python.exe E:/development/sensors_active_monitoring/gpu_analysis_utilities.py
# Memory Temperature [°C] max: 66.0
# Memory Temperature [°C] min: 36.0
# Memory Temperature [°C] mean: 48.37158943265483
# Memory Temperature [°C] median: 50.0
# Memory Temperature [°C] quantile 0.1: 38.0
# Memory Temperature [°C] quantile 0.95: 62.0
#
# Process finished with exit code 0



















