from typing import List, Union
import numpy as np
import pandas as pd
import glob
import os
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(color_codes=True)


def plot_matrix(mat: Union[pd.DataFrame, np.ndarray], fontsz: int, cbar_ticks: List[float] = None):
    """
    :param mat: matrix to plot. If using dataframe, the columns are automatically used as labels. Othereise, matrix is anonymous
    :param fontsz: font size
    :param cbar_ticks: the spacing between cbar ticks. If None, this is set automatically.
    :return:
    """
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    plt.figure(figsize=[8, 8])
    if cbar_ticks is not None:
        ax = sns.heatmap(mat, cmap=cmap, vmin=min(cbar_ticks), vmax=max(cbar_ticks), square=True, linewidths=.5, cbar_kws={"shrink": .5})
        cbar = ax.collections[0].colorbar
        cbar.set_ticks(cbar_ticks)
        cbar.set_ticklabels(cbar_ticks)
    else:
        ax = sns.heatmap(mat, cmap=cmap, vmin=np.min(np.array(mat).ravel()), vmax=np.max(np.array(mat).ravel()), square=True, linewidths=.5, cbar_kws={"shrink": .5})
        cbar = ax.collections[0].colorbar

    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=fontsz)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=fontsz)
    plt.show()


def correlation_matrix(df: pd.DataFrame, fontsz: int = 16, corrThr: float = None):
    """
    :param df: input dataframe. Correlation matrix calculated for all columns
    :param fontsz: font size
    :param toShow: True - plots the figure
    :return:
    """
    # Correlation between numeric variables
    cols_numeric = list(df)
    data_numeric = df[cols_numeric].copy(deep=True)
    corr_mat = data_numeric.corr(method='pearson')
    if corrThr is not None:
        assert corr_mat > 0.0, "corrThr must be a float between [0, 1]"
        corr_mat[corr_mat >= corrThr] = 1.0
        corr_mat[corr_mat <= -corrThr] = -1.0

    cbar_ticks = [round(num, 1) for num in np.linspace(-1, 1, 11, dtype=np.float)]  # rounding corrects for floating point imprecision
    plot_matrix(corr_mat, fontsz=fontsz, cbar_ticks=cbar_ticks)


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
    return sensors_all


if __name__ == '__main__':
    path = r'C:\Users\Shay\Desktop\overclocking\Alien_Isolation_sensors'
    files = get_sensors_files_from_path(path)
    sensors_all = read_and_concat_sensors_data_from_files(files)

    # correlation_matrix(sensors_all, fontsz=8)
    sns.lineplot(data=sensors_all, x="Date", y='GPU Temperature [°C]')
    # fixme: there's a problem with date parsing. the date is completely wrong
    plt.show()


