FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

# ---------- user & locale ----------
ARG USER_ID=1130
ARG GROUP_ID=300
ARG USER_NAME="yyang"

RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime
RUN groupadd -g "${GROUP_ID}" "${USER_NAME}" && useradd -u "${USER_ID}" -m -g "${USER_NAME}" -s /bin/bash "${USER_NAME}"
RUN echo "${USER_NAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Tokyo \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# ---------- system deps (incl. FFmpeg dev for PyAV) ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget ca-certificates \
    build-essential cmake ninja-build pkg-config \
    nasm yasm \
    ffmpeg \
    libavdevice-dev libavfilter-dev libavformat-dev libavcodec-dev \
    libswscale-dev libswresample-dev libavutil-dev \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    vim nano tmux htop unzip zip \
 && rm -rf /var/lib/apt/lists/*

# --- Install Miniconda ---
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
 && bash /tmp/miniconda.sh -b -p /opt/miniconda3 \
 && rm -f /tmp/miniconda.sh
ENV PATH=/opt/miniconda3/bin:$PATH
SHELL ["/bin/bash","-lc"]

# --- Write a system .condarc that removes defaults entirely ---
# This prevents any contact with https://repo.anaconda.com/*
RUN mkdir -p /etc/conda && cat > /etc/conda/.condarc <<'YAML'
channels:
  - conda-forge
channel_priority: strict
default_channels:
  - https://conda.anaconda.org/conda-forge
custom_channels: {}
YAML

# (optional but helpful) also put the same config in root's home
RUN mkdir -p /root && ln -sf /etc/conda/.condarc /root/.condarc

# --- Update conda using only conda-forge (override to be extra safe) ---
RUN conda update -n base -y -c conda-forge --override-channels conda

# --- Create env (again: override channels to avoid implicit defaults) ---
ARG ENV_NAME=animal
RUN conda create -y -n ${ENV_NAME} -c conda-forge --override-channels python=3.9 && \
    conda init bash && echo "conda activate ${ENV_NAME}" >> ~/.bashrc

ENV CONDA_DEFAULT_ENV=${ENV_NAME}
ENV PATH=/opt/miniconda3/envs/${ENV_NAME}/bin:$PATH




# ---------- PyTorch 1.13.0 + cu116 (from PyTorch index) ----------
RUN conda run -n ${ENV_NAME} python -m pip install --upgrade pip setuptools wheel \
 && conda run -n ${ENV_NAME} python -m pip install \
      --extra-index-url https://download.pytorch.org/whl/cu116 \
      torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0+cu116

# ---------- PyAV build prereqs + install av==10.0.0 ----------
# Cython 3 breaks PyAV 10 build; pin Cython<3 and disable build isolation
RUN conda run -n ${ENV_NAME} python -m pip install "cython<3" "setuptools<70" wheel \
 && conda run -n ${ENV_NAME} python -m pip install --no-build-isolation av==10.0.0

# ---------- requirements ----------
WORKDIR /workspace
COPY requirements.txt .

# Clean up problematic lines to avoid duplicates/bogus packages
# - remove exact 'av==10.0.0' (already installed)
# - remove 'cPython==0.0.6' (not a real PyPI package)
RUN sed -i '/^av==10\.0\.0$/d' requirements.txt || true \
 && sed -i '/^cPython==0\.0\.6$/d' requirements.txt || true

# NOTE: if your requirements still have numpy==1.21.6 + mkl-fft==1.3.1, resolver will fail.
# Either bump NumPy to 1.22.4, or pin mkl-fft to 1.3.0 or 1.2.x.
# Example auto-bump (uncomment if you want it enforced in build):
# RUN sed -i 's/^numpy==1\.21\.6$/numpy==1.22.4/' requirements.txt

RUN conda run -n ${ENV_NAME} python -m pip install --no-cache-dir -r requirements.txt

# ---------- mmcv-full 1.6.1 (cu116 + torch1.13.0) ----------
RUN conda run -n ${ENV_NAME} python -m pip install \
  -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.13.0/index.html \
  mmcv-full==1.6.1

# ---------- optional: NVIDIA Apex from source (set APEX=0 to skip) ----------
ARG APEX=0
RUN if [ "$APEX" = "1" ]; then \
      git clone --depth 1 https://github.com/NVIDIA/apex.git /tmp/apex && \
      conda run -n ${ENV_NAME} python -m pip install -v --no-cache-dir \
        --global-option="--cpp_ext" --global-option="--cuda_ext" /tmp/apex || true && \
      rm -rf /tmp/apex ; \
    fi

RUN /bin/bash -c "conda init bash"
RUN echo "conda activate animal" >> /root/.bashrc

# Now mount the actual directory, hopefully
WORKDIR /home/yyang/mnt/workspace/
