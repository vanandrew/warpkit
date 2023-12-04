FROM ubuntu:22.04 as base
LABEL maintainer="Andrew Van <vanandrew@wustl.edu>"

# set noninteractive mode for apt-get
ENV DEBIAN_FRONTEND=noninteractive

# set working directory to /opt
WORKDIR /opt

# get python and other dependencies
RUN apt-get update && \
    apt-get install -y build-essential curl git python3 python3-pip unzip

# get and install Julia via juliaup
FROM base as julia
RUN curl -fsSL https://install.julialang.org | sh -s -- --yes --default-channel 1.9.4 && \
    mkdir -p /opt/julia/ && cp -r /root/.julia/juliaup/*/* /opt/julia/

# final image
FROM base as final

# copy over julia
COPY --from=julia /opt/julia/ /opt/julia/
ENV PATH=/opt/julia/bin:${PATH}
# add libjulia to ldconfig
RUN echo "/opt/julia/lib" >> /etc/ld.so.conf.d/julia.conf && ldconfig

# get and install warpkit
ADD . /opt/warpkit

# install warpkit
RUN cd /opt/warpkit && pip3 install -U pip && pip3 install -e ./[dev] -v --config-settings editable_mode=strict

# test warpkit
RUN cd /opt/warpkit/tests/data/ && ./download_bids_testdata.sh && cd /opt/warpkit/ && pytest -s -v

# set medic script as entrypoint
ENTRYPOINT ["medic"]
