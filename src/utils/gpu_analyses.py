import streamlit as st

from utils.analytical_utils import read_sensors_data_from_file, add_nan_values_on_time_gap, calc_prefetch_cap_reasons, \
    PERF_CAP_REASON
from utils.visualization_utils import create_multiple_sensor_graphs, calc_and_plot_reasons_correlation


def main(file):
    sensors_all = read_sensors_data_from_file(file)
    cols_to_set = [col for col in sensors_all if col != 'Date']
    sensors_nan_gaps = add_nan_values_on_time_gap(sensors_all, 'Date', cols_to_set)

    # TODO: add verification that these columns exist.
    default_columns = ['CPU Temperature [°C]', 'GPU Temperature [°C]', 'Memory Temperature [°C]', 'Hot Spot [°C]',
                       'GPU Chip Power Draw [W]', 'GPU Load [%]', 'Fan 1 Speed (%) [%]', 'System Memory Used [MB]']

    selected_columns = st.multiselect('Select up to 8 options:', cols_to_set,
                                      max_selections=8, default=default_columns)
    st.session_state.selected_options = selected_columns

    create_multiple_sensor_graphs(selected_columns, sensors_nan_gaps)

    reasons_df = calc_prefetch_cap_reasons(sensors_all)
    col1, col2 = st.columns(2)  # Split the app layout into two columns
    with col1:  # Use the first column for the chart
        st.write("### Occurrences of Reasons")
        st.bar_chart(reasons_df['Count'], width=0.5)  # Adjust width to 50%

    pref_cat_reasons_correlation_matrix = calc_and_plot_reasons_correlation(sensors_all['Reasons'],
                                                                            list(PERF_CAP_REASON.values()))
    with col2:
        st.write("### Correlation between Reasons")
        st.write(pref_cat_reasons_correlation_matrix)





