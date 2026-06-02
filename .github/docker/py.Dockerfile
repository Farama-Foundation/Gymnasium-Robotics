# A Dockerfile that sets up a full gymnasium-robotics install with test dependencies
ARG BASE_PYTHON_VERSION=3.10
FROM python:$BASE_PYTHON_VERSION
ARG BASE_PYTHON_VERSION

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get -y update && apt-get install -y unzip libglu1-mesa-dev libgl1-mesa-dev libosmesa6-dev xvfb patchelf ffmpeg cmake swig

# Download MuJoCo 2.1.0 for the deprecated mujoco-py tests.
RUN if [[ "$BASE_PYTHON_VERSION" == 3.14* ]]; then \
        echo "Skipping MuJoCo 2.1.0 download on Python 3.14"; \
    else \
        mkdir /root/.mujoco \
        && cd /root/.mujoco \
        && wget -qO- 'https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz' | tar -xzvf -; \
    fi

ENV LD_LIBRARY_PATH="/root/.mujoco/mujoco210/bin"
# Use OSMesa for headless OpenGL rendering in MuJoCo
ENV MUJOCO_GL="osmesa"
# mujoco-py does JIT compilation at runtime, so CFLAGS must persist as an env var
# GCC 14+ treats pointer type mismatches and implicit declarations as hard errors
# -fpermissive doesn't work for C, so we must explicitly disable these
ENV CFLAGS="-Wno-incompatible-pointer-types -Wno-implicit-function-declaration -Wno-int-conversion -w"

# Build mujoco-py from source. PyPI installs wheel packages and Cython won't recompile old file versions in the GitHub Actions CI.
# Thus generating the following error https://github.com/cython/cython/pull/4428.
# NOTE: mujoco-py requires:
#   - numpy<2.0 due to incompatible C API changes
#   - setuptools for distutils (removed in Python 3.12)
# Python 3.14 is tested without mujoco-py because the deprecated mujoco-py stack
# depends on numpy<2.0, which does not provide Python 3.14 wheels.
RUN if [[ "$BASE_PYTHON_VERSION" == 3.14* ]]; then \
        echo "Skipping deprecated mujoco-py setup on Python 3.14"; \
    else \
        pip install "numpy<2.0" setuptools \
        && git clone https://github.com/Kallinteris-Andreas/mujoco-py.git \
        && cd mujoco-py \
        && pip install -e . \
        && python -c "import mujoco_py"; \
    fi

COPY . /usr/local/gymnasium-robotics/
WORKDIR /usr/local/gymnasium-robotics/

RUN pip install "gymnasium @ git+https://github.com/Farama-Foundation/Gymnasium.git@main" .[testing] --no-cache-dir

ENTRYPOINT ["/usr/local/gymnasium-robotics/.github/docker/entrypoint"]
