FROM ubuntu:24.04 AS base
LABEL maintainer="Andrew Van <vanandrew@wustl.edu>"

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /opt

# build deps (python comes from uv)
RUN apt-get update && \
    apt-get install -y build-essential cmake curl git unzip ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# install uv (static binary)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# get and install warpkit
ADD . /opt/warpkit

# sync project (installs python 3.11, deps, and builds the CMake extension)
ENV UV_PROJECT_ENVIRONMENT=/opt/warpkit/.venv
ENV UV_PYTHON_PREFERENCE=only-managed
RUN cd /opt/warpkit && uv sync --group dev --config-setting editable_mode=strict -v

# put the project venv on PATH so `medic` and friends resolve
ENV PATH=/opt/warpkit/.venv/bin:${PATH}

# test warpkit
RUN cd /opt/warpkit && uv run pytest -s -v

# set medic script as entrypoint
ENTRYPOINT ["medic"]
