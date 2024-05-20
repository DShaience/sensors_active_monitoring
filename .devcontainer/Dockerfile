FROM mambaorg/micromamba:0.15.3
USER root
# RUN mkdir /opt/demo_azure
# RUN chmod -R 777 /opt/demo_azure
# WORKDIR /opt/demo_azure
RUN  mkdir /app
RUN chmod -R 777 /app
WORKDIR /app
USER micromamba
COPY ./src/app_conda_env.yml /opt/demo_azure/app_conda_env.yml
COPY ./src ./src

EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

COPY ./src/app_conda_env.yml /src/app_conda_env.yml
WORKDIR /app/src
RUN micromamba install -y -n base -f app_conda_env.yml && micromamba clean --all --yes
# COPY ./src/run_app.sh run_app.sh
USER root
# RUN chmod a+x run_app.sh
RUN chmod a+x run.py
RUN chmod a+x run_app.sh

CMD ["streamlit", "run", "run.py", "--server.port=8051", "--server.address=0.0.0.0", "--server.enableWebsocketCompression=false"]

# CMD ["streamlit", "run", "app.py", "--server.port=8051", "--server.address=0.0.0.0", "--server.enableWebsocketCompression=false"]

# CMD ["./run_app.sh"]
# ENTRYPOINT ["streamlit", "run", "run.py", "--server.port=8501", "--server.address=0.0.0.0"]
# ENTRYPOINT ["./run_app.sh"]
