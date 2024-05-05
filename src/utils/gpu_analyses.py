import datetime as dt

import streamlit as st

from utils.analytical_utils import read_sensors_data_from_file, add_nan_values_on_time_gap, calc_prefetch_cap_reasons, get_default_columns
from utils.consts import DEFAULT_COLUMNS, PERF_CAP_REASON
from utils.visualization_utils import create_multiple_sensor_graphs, calc_and_plot_reasons_correlation


def sensor_graphs(file, date_time_delta: dt.timedelta):
    sensors_all = read_sensors_data_from_file(file)
    cols_to_set = [col for col in sensors_all if col != 'Date']
    sensors_nan_gaps = add_nan_values_on_time_gap(sensors_all, 'Date', cols_to_set)

    if date_time_delta is not None:
        max_date = sensors_nan_gaps['Date'].max()
        sensor_start_time = max_date - date_time_delta
        sensors_nan_gaps = sensors_nan_gaps[sensors_nan_gaps['Date'] >= sensor_start_time]
        if sensors_nan_gaps.empty:
            st.warning("Time filter shows no records in this time-frame. Please select another time-frame")
            st.stop()  # Stop further execution if there's no data

    default_columns = get_default_columns(DEFAULT_COLUMNS, cols_to_set)

    selected_columns = st.multiselect('Select up to 8 options:', cols_to_set,
                                      max_selections=8, default=default_columns)
    st.session_state.selected_options = selected_columns

    create_multiple_sensor_graphs(selected_columns, sensors_nan_gaps)

    # FIXME: make it so that reasons are also affected by time-frames
    # Create PerfCap reason matrix only if exists
    if 'PerfCap Reason []' in cols_to_set:
        reasons_df = calc_prefetch_cap_reasons(sensors_all)
        col1, col2 = st.columns(2)  # Split the app layout into two columns
        with col1:  # Use the first column for the chart
            st.write("### GPU Prefetch-Cap Reasons (hist)")
            st.bar_chart(reasons_df['Count'], width=0.5)  # Adjust width to 50%

        pref_cat_reasons_correlation_matrix = calc_and_plot_reasons_correlation(sensors_all['Reasons'],
                                                                                list(PERF_CAP_REASON.values()))
        with col2:
            st.write("### GPU Prefetch-Cap Reasons correlation matrix")
            st.write(pref_cat_reasons_correlation_matrix)





