import streamlit

import streamlit.web.cli as stcli
import os, sys
from utils.consts import DEFAULT_SERVER_PORT


def resolve_path(path):
    resolved_path = os.path.abspath(os.path.join(os.getcwd(), path))
    return resolved_path


if __name__ == "__main__":
    sys.argv = [
        "streamlit",
        "run",
        resolve_path("webapp/app.py"),
        "--global.developmentMode=false",
        "--browser.gatherUsageStats=false",
        f"--server.port={DEFAULT_SERVER_PORT}",
    ]
    sys.exit(stcli.main())


## REVIEW THIS FOR CREATING A STREAMLIT EXECUTABLE APP
# https://www.google.com/search?client=firefox-b-d&q=packaging+streamlit+app+as+executable
# https://ploomber.io/blog/streamlit_exe/

# With App Service
# https://towardsdatascience.com/beginner-guide-to-streamlit-deployment-on-azure-f6618eee1ba9

# Check this out for azure/github/copilot
# https://docs.github.com/en/billing/managing-the-plan-for-your-github-account/connecting-an-azure-subscription
