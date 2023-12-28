import seaborn as sns
import streamlit as st

from utils.analytical_utils import read_sensors_data_from_file, add_nan_values_on_time_gap, calc_prefetch_cap_reasons, \
    PERF_CAP_REASON
from utils.visualization_utils import create_multiple_sensor_graphs, plot_reasons_bar_chart, \
    plot_reasons_correlation_matrix

sns.set(color_codes=True)


def main(file):
    sensors_all = read_sensors_data_from_file(file)
    cols_to_set = [col for col in sensors_all if col != 'Date']
    sensors_nan_gaps = add_nan_values_on_time_gap(sensors_all, 'Date', cols_to_set)

    # Plot sensors graphs
    # TODO: add verification that these columns exist.
    default_columns = ['CPU Temperature [°C]', 'GPU Temperature [°C]', 'Memory Temperature [°C]', 'Hot Spot [°C]',
                       'GPU Chip Power Draw [W]', 'GPU Load [%]', 'Fan 1 Speed (%) [%]', 'System Memory Used [MB]']

    selected_columns = st.multiselect('Select up to 8 options:', cols_to_set,
                                      max_selections=8, default=default_columns)
    st.session_state.selected_options = selected_columns
    features_list, sensor_graphs = create_multiple_sensor_graphs(selected_columns, sensors_nan_gaps)
    st.markdown(
        f'<div style="width: 50%;"><img src="data:image/png;base64,{sensor_graphs}" alt="plot"></div><br>',
        unsafe_allow_html=True
    )

    # Plot GPU Prefetch cap reason
    reasons_df = calc_prefetch_cap_reasons(sensors_all)
    reasons_graph = plot_reasons_bar_chart(reasons_df)
    st.markdown(
        f'<div style="width: 50%;"><img src="data:image/png;base64,{reasons_graph}" alt="plot"></div>',
        unsafe_allow_html=True
    )

    pref_cat_reasons_correlation_matrix = plot_reasons_correlation_matrix(sensors_all, list(PERF_CAP_REASON.values()))
    st.markdown(
        f'<div style="width: 50%;"><img src="data:image/png;base64,{pref_cat_reasons_correlation_matrix}" alt="plot"></div>',
        unsafe_allow_html=True
    )



