# A Dockerfile that sets up a full gymnasium-robotics install with test dependencies
ARG PYTHON_VERSION
FROM python:$PYTHON_VERSION

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt update && apt install software-properties-common && add-apt-repository ppa:ubuntu-toolchain-r/test && apt-get -y update && apt-get install -y unzip libglu1-mesa-dev libgl1-mesa-dev libosmesa6-dev xvfb patchelf ffmpeg cmake swig gcc-9 g++-9

# Download mujoco
RUN mkdir /root/.mujoco \
    && cd /root/.mujoco \
    && wget -qO- 'https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz' | tar -xzvf -

ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin"

# Build mujoco-py from source. Pypi installs wheel packages and Cython won't recompile old file versions in the Github Actions CI.
# Thus generating the following error https://github.com/cython/cython/pull/4428
RUN git clone https://github.com/Kallinteris-Andreas/mujoco-py\
    && cd mujoco-py \
    && CC=gcc-9 CXX=g++-9 pip install -e .

COPY . /usr/local/gymnasium-robotics/
WORKDIR /usr/local/gymnasium-robotics/

RUN pip install .[testing] --no-cache-dir

ENTRYPOINT ["/usr/local/gymnasium-robotics/.github/docker/entrypoint"]
