from typing import Union, List

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def plot_matrix(mat: Union[pd.DataFrame, np.ndarray], font_size: int, cbar_ticks: List[float] = None):
    """
    :param mat: matrix to plot. If using dataframe, the columns are automatically used as labels. Otherwise, matrix is anonymous
    :param font_size: font size
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

    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=font_size)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=font_size)
    plt.show()


def correlation_matrix(df: pd.DataFrame, font_size: int = 16, correlation_thr: float = None):
    """
    :param df: input dataframe. Correlation matrix calculated for all columns
    :param font_size: font size
    :param correlation_thr:
    :return:
    """
    # Correlation between numeric variables
    cols_numeric = list(df)
    data_numeric = df[cols_numeric].copy(deep=True)
    corr_mat = data_numeric.corr(method='pearson')
    if correlation_thr is not None:
        assert corr_mat > 0.0, "corrThr must be a float between [0, 1]"
        corr_mat[corr_mat >= correlation_thr] = 1.0
        corr_mat[corr_mat <= -correlation_thr] = -1.0

    cbar_ticks = [round(num, 1) for num in np.linspace(-1, 1, 11, dtype=np.float)]  # rounding corrects for floating point imprecision
    plot_matrix(corr_mat, font_size=font_size, cbar_ticks=cbar_ticks)


def plot_reasons_bar_chart(reasons_df: pd.DataFrame):
    # Create a bar chart
    reasons_df.plot(kind='barh', color='skyblue', figsize=(14, 6))
    plt.title('Occurrences of Reasons')
    plt.xlabel('Count')
    plt.ylabel('Reasons')
    plt.grid(axis='x')
    plt.tight_layout()
    plt.show()


def plot_reasons_correlation_matrix(df: pd.DataFrame, reasons: list[str]):
    # Create a matrix of zeros with reasons as columns and index
    reasons_matrix = pd.DataFrame(0, index=reasons, columns=reasons)
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
