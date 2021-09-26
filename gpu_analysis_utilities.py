from typing import List
import numpy as np
import pandas as pd
import glob
import os
from matplotlib import pyplot as plt
import seaborn as sns
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


if __name__ == '__main__':
    path = r'C:\Users\Shay\Desktop\Desktop items\overclocking\Sensors_data'
    files = get_sensors_files_from_path(path)
    sensors_all = read_and_concat_sensors_data_from_files(files)

    cols_to_set = [col for col in sensors_all if col != 'Date']
    sensors_nan_gaps = add_nan_values_on_time_gap(sensors_all, 'Date', cols_to_set)
    # correlation_matrix(sensors_nan_gaps, fontsz=8)
    # [col for col in cols_to_set if 'gpu' in col.lower()]

    # target_col = 'GPU Temperature [°C]'
    # target_col = 'Power Consumption (%) [% TDP]'
    target_col = 'CPU Temperature [°C]'
    # target_col = 'GPU Load [%]'
    # target_col = 'Board Power Draw [W]'
    # target_col = 'Hot Spot [°C]'
    # target_col = 'Fan Speed (RPM) [RPM]'
    # target_col = 'Fan Speed (%) [%]'
    # target_col = 'GPU Chip Power Draw [W]'
    target_col_values_no_nan = sensors_nan_gaps.loc[~sensors_nan_gaps[target_col].isna(), target_col].values
    print(f"{target_col} max: {np.max(target_col_values_no_nan)}")
    print(f"{target_col} min: {np.min(target_col_values_no_nan)}")
    print(f"{target_col} mean: {np.mean(target_col_values_no_nan)}")
    print(f"{target_col} median: {np.median(target_col_values_no_nan)}")
    print(f"{target_col} quantile 0.1: {np.quantile(target_col_values_no_nan, 0.1)}")
    print(f"{target_col} quantile 0.95: {np.quantile(target_col_values_no_nan, 0.95)}")
    sensors_nan_gaps[target_col].plot()
    plt.show()






















