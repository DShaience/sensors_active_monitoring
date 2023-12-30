import streamlit

import streamlit.web.cli as stcli
import os, sys


def resolve_path(path):
    resolved_path = os.path.abspath(os.path.join(os.getcwd(), path))
    return resolved_path


if __name__ == "__main__":
    sys.argv = [
        "streamlit",
        "run",
        resolve_path("webapp/app.py"),
        "--global.developmentMode=false",
    ]
    sys.exit(stcli.main())



## REVIEW THIS FOR CREATING A STREAMLIT EXECUTABLE APP
# https://www.google.com/search?client=firefox-b-d&q=packaging+streamlit+app+as+executable
# https://ploomber.io/blog/streamlit_exe/
