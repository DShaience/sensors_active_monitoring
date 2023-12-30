import pandas as pd

from utils.analytical_utils import calc_statistics
import streamlit as st



# TODO: add documentation
def create_multiple_sensor_graphs(selected_columns, sensors_nan_gaps):
    features_list = []
    col1, col2 = st.columns(2)  # Creating two columns layout

    half_len = len(selected_columns) // 2
    for i, target_col in enumerate(selected_columns):
        if i < half_len:
            with col1:  # Place charts in the first column
                st.write(f"### {target_col}")
                data = sensors_nan_gaps.loc[~sensors_nan_gaps[target_col].isna(), target_col].values
                st.line_chart(data)
                features_list.append(calc_statistics(data, target_col.replace(' ', '_')))
        else:
            with col2:  # Place charts in the second column
                st.write(f"### {target_col}")
                data = sensors_nan_gaps.loc[~sensors_nan_gaps[target_col].isna(), target_col].values
                st.line_chart(data)
                features_list.append(calc_statistics(data, target_col.replace(' ', '_')))

    return features_list


def plot_reasons_bar_chart(reasons_df: pd.DataFrame):
    st.write("### Occurrences of Reasons")

    col1, col2 = st.columns(2)  # Split the app layout into two columns

    with col1:  # Use the first column for the chart
        st.bar_chart(reasons_df['Count'], width=0.5)  # Adjust width to 50%

    return


def calc_and_plot_reasons_correlation(reasons: pd.Series, reasons_types: list[str]):
    # Create a matrix of zeros with reasons as columns and index
    reasons_matrix = pd.DataFrame(0, index=reasons_types, columns=reasons_types)

    # Update the matrix with counts of reasons appearing together
    for reasons_list in reasons:
        for i in range(len(reasons_list)):
            for j in range(i + 1, len(reasons_list)):
                reasons_matrix.loc[reasons_list[i], reasons_list[j]] += 1
                reasons_matrix.loc[reasons_list[j], reasons_list[i]] += 1

    # Display the correlation matrix as a table
    st.write("### Correlation between Reasons")
    st.write(reasons_matrix)

    return reasons_matrix
