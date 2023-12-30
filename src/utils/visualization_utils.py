import base64
import io

import pandas as pd
from matplotlib import pyplot as plt

from utils.analytical_utils import calc_statistics


def plot_reasons_bar_chart(reasons_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 6))

    reasons = reasons_df.index
    counts = reasons_df['Count']
    ax.barh(reasons, counts, color='skyblue')
    ax.set_title('Occurrences of Reasons')
    ax.set_xlabel('Count')
    ax.set_ylabel('Reasons')
    ax.invert_yaxis()
    ax.grid(axis='x')
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
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(24, 12))
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
    fig, ax = plt.subplots(figsize=(10, 8))

    # Generate a heatmap for the correlation matrix using Matplotlib
    heatmap = ax.imshow(reasons_matrix, cmap='coolwarm')

    # Set ticks and labels for both axes
    n_corr = len(reasons_matrix.columns)
    ax.set_xticks(range(n_corr))
    ax.set_yticks(range(n_corr))
    ax.set_xticklabels(reasons_matrix.columns)
    ax.set_yticklabels(reasons_matrix.index)

    # Shift the ticks to align with the center of the cells
    ax.set_xticks([i + 0.5 for i in range(n_corr)], minor=True)
    ax.set_yticks([i + 0.5 for i in range(n_corr)], minor=True)

    # Display gridlines
    ax.grid(which='major', color='white', linestyle='-', linewidth=0)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=1.5)

    # Display values on the heatmap
    for i in range(len(reasons_types)):
        for j in range(len(reasons_types)):
            text = ax.text(j, i, reasons_matrix.iloc[i, j], ha='center', va='center', color='black')

    # Set title and labels
    ax.set_title('Correlation between Reasons')
    ax.set_xlabel('Reasons')
    ax.set_ylabel('Reasons')

    # Show color bar
    cbar = ax.figure.colorbar(heatmap, ax=ax)
    cbar.ax.set_ylabel("Counts", rotation=-90, va="bottom")

    # Rotate ticks on x-axis for better readability
    plt.xticks(rotation=90)

    # Adjust layout and display the plot
    plt.tight_layout()
    plot_image = fig_to_img(fig)
    return plot_image

