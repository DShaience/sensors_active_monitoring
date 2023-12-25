from typing import List
import numpy as np
import pandas as pd
import glob
import os
from matplotlib import pyplot as plt
import seaborn as sns
from collections import OrderedDict
sns.set(color_codes=True)


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


def get_sensors_files_from_path(path: str):
    files = list(glob.glob(path + os.sep + '*.txt', recursive=False))
    return files


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


def plot_reasons_bar_chart(reasons_df: pd.DataFrame):
    # Create a bar chart
    reasons_df.plot(kind='barh', color='skyblue', figsize=(14, 6))
    plt.title('Occurrences of Reasons')
    plt.xlabel('Count')
    plt.ylabel('Reasons')
    plt.grid(axis='x')
    plt.tight_layout()
    plt.show()


def plot_reasons_correlation_matrix(df: pd.DataFrame):
    global i
    # Create a matrix of zeros with reasons as columns and index
    reasons_matrix = pd.DataFrame(0, index=reasons_df.index, columns=reasons_df.index)
    # Update the matrix with counts of reasons appearing together
    for reasons in df['Reasons']:
        for i in range(len(reasons)):
            for j in range(i + 1, len(reasons)):
                reasons_matrix.loc[reasons[i], reasons[j]] += 1
                reasons_matrix.loc[reasons[j], reasons[i]] += 1
    # Generate a heatmap for the correlation matrix
    plt.figure(figsize=(10, 10))
    sns.heatmap(reasons_matrix, annot=True, cmap='coolwarm', fmt='d')
    plt.title('Correlation between Reasons')
    plt.xlabel('Reasons')
    plt.ylabel('Reasons')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # path = r'C:\Users\Shay\Desktop\Desktop items\overclocking\Sensors_data'
    # path = r'E:\Backups\Desktop\Desktop items\overclocking\Sensors_data'
    path = r'C:\Users\shayt\OneDrive\Desktop\Sensors'
    files = get_sensors_files_from_path(path)
    # files = [file for file in files if "RTX 3070 Ti Asus 04 CPU-OC_01.txt" not in file]
    # files = [file for file in files if "Baseline3.txt" in file]
    # files = [file for file in files if "Psychonauts2" in file]
    # files = [file for file in files if "Requiem" in file]
    # files = [file for file in files if "Origins" in file]
    file = [file for file in files if "Morales" in file][0]
    # file = [file for file in files if "Unigine Heaven Benchmark" in file][0]
    sensors_all = read_sensors_data_from_file(file)

    cols_to_set = [col for col in sensors_all if col != 'Date']
    sensors_nan_gaps = add_nan_values_on_time_gap(sensors_all, 'Date', cols_to_set)

    target_columns = ['CPU Temperature [°C]', 'GPU Temperature [°C]', 'Memory Temperature [°C]', 'Hot Spot [°C]',
                      'GPU Chip Power Draw [W]',  'GPU Load [%]', 'Fan 1 Speed (%) [%]', 'System Memory Used [MB]']

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

    sensors_all['PerfCap Reason []'] = sensors_all['PerfCap Reason []'].fillna(value=0).astype(int)
    sensors_all['Reasons'] = sensors_all['PerfCap Reason []'].apply(extract_reasons)

    # Flatten the list of reasons
    reasons_list = [reason for sublist in sensors_all['Reasons'] for reason in sublist]

    from collections import Counter
    # Count occurrences of each reason
    reasons_count = Counter(reasons_list)

    # Convert the Counter to a DataFrame for plotting
    reasons_df = pd.DataFrame.from_dict(reasons_count, orient='index', columns=['Count'])
    reasons_df.sort_values(by='Count', ascending=True, inplace=True)

    plot_reasons_bar_chart(reasons_df)

    PLOT_CORRELATION = False
    if PLOT_CORRELATION:
        plot_reasons_correlation_matrix(sensors_all)

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

