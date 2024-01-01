import os

import streamlit as st
from utils.gpu_analyses import sensor_graphs

from datetime import datetime, timedelta

TIMEFRAMES = {
    'Last 1 Hour': timedelta(hours=1),
    'Last 2 Hours': timedelta(hours=2),
    'Last 4 Hours': timedelta(hours=4),
    'Last 24 Hours': timedelta(days=1),
    'All': None
}


def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)


def app():
    st.set_page_config(layout="wide")  # Optional: Set page configuration
    st._is_running_with_streamlit = True  # Set a flag for Streamlit compatibility
    st.title('GPU-Z Logs Visualizer')

    col1, col2, col3, col4, col5 = st.columns(5)
    # Adjust the file uploader width in the second column
    with col1:
        uploaded_file = st.file_uploader("Upload a file", type=['txt'])

    with col2:
        time_delta_option = st.selectbox(
            'Select timeframe to read the corresponding time window from file',
            list(TIMEFRAMES.keys()),
            index=len(TIMEFRAMES) - 1  # Set the default index to the last option ('All')
        )
        st.write('You selected:', time_delta_option)

    if uploaded_file is not None:
        sensor_graphs(uploaded_file, TIMEFRAMES[time_delta_option])


if __name__ == '__main__':
    app()


