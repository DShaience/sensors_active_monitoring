import sys
sys.path.append('../')  # Add parent directory to Python path

import os

import pandas as pd
import numpy as np

import streamlit as st
# from streamlit_file_browser import st_file_browser
from gpu_analysis_run import main


def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)


if __name__ == '__main__':
    st.title('Uber pickups in NYC')

    uploaded_file = st.file_uploader("Upload a file", type=['txt', 'csv', 'pdf'])

    if uploaded_file is not None:
        # Process the uploaded file
        # file_contents = uploaded_file.getvalue()

        main(uploaded_file)


    # filename = file_selector()
    # st.write('You selected `%s`' % filename)
    # st.header('Default Options')
    # event = st_file_browser("example_artifacts", key='A')
    # st.write(event)
