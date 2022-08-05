# BUILD IMAGE: docker build -t soft/teacher:v1 .
# START CONTAINER: docker run -v /raid/utku/:/workspace -t pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime
# Docker volume (~mount):
# You can also copy with ADD. This copies, not good ADD /raid/utku /workspace 
# Docker cp to copy from local to docker running container.
# Not suggested: ENV DEBIAN_FRONTEND=noninteractive 

# ARG PYTORCH_VERSION="1.9.0"
# ARG CUDA_VERSION="10.2"
# ARG CUDNN_VERSION="7"
# # FROM nvidia/cuda:11.3.1-base-ubuntu16.04
# # BASE IMAGE
# FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
# # FROM pytorch/pytorch:${PYTORCH_VERSION}-cuda${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel
# # Create workdir
# # This did not work: COPY . /workspace/

# # Force cuda do not know exactly why.
# ENV FORCE_CUDA="1"

# PYTHON DEPENDENCIES
# ENV PYTHON_VERSION=3.6
# ENV MMDETECTION_VERSION=2.16.0+fe46ffe
# ENV MMCV_VERSION=1.3.9
# ENV WANDB=0.10.31

# Apt installations.  python3-opencv
#&& apt-get clean \
# && rm -rf /var/lib/apt/lists/*

# RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
#     gcc git tmux nano make
# WORKDIR /workspace
# RUN make install

# RUN apt-get update
# RUN apt-get install software-properties-common -y
# RUN add-apt-repository ppa:deadsnakes/ppa
# RUN apt-get update
# RUN apt-get install python3.6 -y 
# RUN apt-get install python3.6-dev
# RUN apt-get install python3.6-pip

# Install python packages ==${MMDETECTION_VERSION}
# RUN pip3 install mmcv==${MMCV_VERSION} wandb==${WANDB}
# Install mmdetections
# RUN git clone https://github.com/open-mmlab/mmdetection.git /mmdetection
# WORKDIR /mmdetection
# RUN pip3 install -r requirements/build.txt
# RUN pip3 install --no-cache-dir -e .

# install requirments.














FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
ENV PROJECT=soft_teacher
ENV PYTORCH_VERSION=1.9.0
# ARG CUDNN_VERSION="7"
ENV CUDNN_VERSION=7.6.5.32-1+cuda10.2
ENV NCCL_VERSION=2.5.6-1+cuda10.2
# apt-cache policy libnccl2
# ENV DEBIAN_FRONTEND=noninteractive

ARG python=3.6
ENV PYTHON_VERSION=${python}

SHELL ["/bin/bash", "-cu"]

RUN apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
    build-essential \
    cmake \
    g++-4.8 \
    git \
    curl \
    docker.io \
    vim \
    wget \
    ca-certificates \
    libcudnn7=${CUDNN_VERSION} \
    libnccl2=${NCCL_VERSION} \
    libnccl-dev=${NCCL_VERSION} \
    libjpeg-dev \
    libpng-dev \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python3-tk \
    librdmacm1 \
    libibverbs1 \
    ibverbs-providers \
    libgtk2.0-dev \
    unzip \
    bzip2 \
    htop \
    gnuplot \
    ffmpeg


# # Install latest CMake
# RUN apt-get remove -y --purge --auto-remove cmake
# RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
# RUN apt-get install -y software-properties-common
# RUN apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'
# RUN apt-get update
# RUN apt-get install -y cmake
# RUN apt-get install -y kitware-archive-keyring
# RUN rm /etc/apt/trusted.gpg.d/kitware.gpg
# RUN apt-get update


# # Instal Python and pip
# RUN if [[ "${PYTHON_VERSION}" == "3.6" ]]; then \
#     apt-get install -y python${PYTHON_VERSION}-distutils; \
#     fi

# RUN ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python

# RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
#     python get-pip.py && \
#     rm get-pip.py

# # Install PyTorch
# RUN pip install torch==${PYTORCH_VERSION} 

# RUN pip install \
#     cython \
#     numpy \
#     pillow \
#     pillow-simd \
#     opencv-python \
#     opencv-contrib-python \
#     opencv-python-headless \
#     numba \
#     tqdm