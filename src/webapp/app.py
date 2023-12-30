import sys
sys.path.append('../')  # Add parent directory to Python path

import os

import streamlit as st
from utils.gpu_analyses import main


def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)


def app():
    st.set_page_config(layout="wide")  # Optional: Set page configuration
    st._is_running_with_streamlit = True  # Set a flag for Streamlit compatibility
    st.title('GPU-Z Logs Visualizer')

    col1, col2 = st.columns([1, 5])
    # Adjust the file uploader width in the second column
    with col1:
        uploaded_file = st.file_uploader("Upload a file", type=['txt'])

    if uploaded_file is not None:
        # Process the uploaded file
        main(uploaded_file)


if __name__ == '__main__':
    app()


