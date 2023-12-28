import subprocess


def run_streamlit_app(script_path):
    subprocess.run(["streamlit", "run", script_path])


if __name__ == "__main__":
    app_script = "app.py"  # Replace with your Streamlit app script path
    run_streamlit_app(app_script)
