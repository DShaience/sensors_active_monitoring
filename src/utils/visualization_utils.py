import base64
import io

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from utils.analytical_utils import calc_statistics


def plot_reasons_bar_chart(reasons_df: pd.DataFrame):
    # Create figure and axes objects
    fig, ax = plt.subplots(figsize=(12, 6))

    # Extracting data from DataFrame
    reasons = reasons_df.index
    counts = reasons_df['Count']

    # Plot the bar chart using Matplotlib
    ax.barh(reasons, counts, color='skyblue')

    # Set title and labels
    ax.set_title('Occurrences of Reasons')
    ax.set_xlabel('Count')
    ax.set_ylabel('Reasons')

    # Invert y-axis to display reasons from top to bottom
    ax.invert_yaxis()

    # Set grid
    ax.grid(axis='x')

    # Adjust layout and display the plot
    plt.tight_layout()
    plot_image = fig_to_img(fig)

    return plot_image


def fig_to_img(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def create_multiple_sensor_graphs(selected_columns, sensors_nan_gaps):
    n_rows = 4
    n_cols = max(1, int(round(len(selected_columns) / n_rows + 0.5, 0)))
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(12, 8))
    features_list = []
    for i in range(len(selected_columns)):
        target_col = selected_columns[i]
        ax = axes.flatten()[i]
        data = sensors_nan_gaps.loc[~sensors_nan_gaps[target_col].isna(), target_col].values
        ax.plot(range(len(data)), data)
        ax.set_title(target_col)
        features_list.append(calc_statistics(data, target_col.replace(' ', '_')))
    if n_cols * n_rows > len(selected_columns):
        for ax in axes.flatten()[len(selected_columns):]:
            ax.remove()
    plt.tight_layout()
    plot_image = fig_to_img(fig)
    return features_list, plot_image


def calc_and_plot_reasons_correlation(reasons: pd.Series, reasons_types: list[str]):
    # Create a matrix of zeros with reasons as columns and index
    reasons_matrix = pd.DataFrame(0, index=reasons_types, columns=reasons_types)

    # Update the matrix with counts of reasons appearing together
    for reasons_list in reasons:
        for i in range(len(reasons_list)):
            for j in range(i + 1, len(reasons_list)):
                reasons_matrix.loc[reasons_list[i], reasons_list[j]] += 1
                reasons_matrix.loc[reasons_list[j], reasons_list[i]] += 1

    # Create figure and axes objects
    fig, ax = plt.subplots(figsize=(12, 10))

    # Generate a heatmap for the correlation matrix using Seaborn
    sns.heatmap(reasons_matrix, annot=True, cmap='coolwarm', fmt='d', ax=ax)

    # Set title and labels
    ax.set_title('Correlation between Reasons')
    ax.set_xlabel('Reasons')
    ax.set_ylabel('Reasons')

    # Adjust layout and display the plot
    plt.tight_layout()
    plot_image = fig_to_img(fig)
    return plot_image

