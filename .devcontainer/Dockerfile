FROM mambaorg/micromamba:0.15.3
USER root
# RUN mkdir /opt/demo_azure
# RUN chmod -R 777 /opt/demo_azure
# WORKDIR /opt/demo_azure
USER micromamba
COPY ./src/app_conda_env.yml /opt/demo_azure/app_conda_env.yml
COPY ./src src

COPY ./src/app_conda_env.yml /src/app_conda_env.yml
WORKDIR /src
RUN micromamba install -y -n base -f app_conda_env.yml && micromamba clean --all --yes
COPY ./src/run_try.sh run_try.sh
USER root
RUN chmod a+x run_try.sh
# CMD ["./run_try.sh"]
