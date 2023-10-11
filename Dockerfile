FROM ubuntu:22.04 as base
LABEL maintainer="Andrew Van <vanandrew@wustl.edu>"

# set noninteractive mode for apt-get
ENV DEBIAN_FRONTEND=noninteractive

# set working directory to /opt
WORKDIR /opt

# get python and other dependencies
RUN apt-get update && \
    apt-get install -y build-essential wget git python3 python3-pip unzip

# get and install Julia
FROM base as julia
RUN wget https://julialang-s3.julialang.org/bin/linux/x64/1.8/julia-1.8.3-linux-x86_64.tar.gz && \
    tar -xzf julia-1.8.3-linux-x86_64.tar.gz && \
    rm julia-1.8.3-linux-x86_64.tar.gz

# final image
FROM base as final

# copy over julia
COPY --from=julia /opt/julia-1.8.3/ /opt/julia/
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
