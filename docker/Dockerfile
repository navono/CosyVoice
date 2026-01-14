FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ARG VENV_NAME="cosyvoice"
ENV VENV=$VENV_NAME
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_CONFIG_FILE=/dev/null
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
SHELL ["/bin/bash", "--login", "-c"]

# Use Aliyun Ubuntu mirror (more reliable in CN/WSL2 mirrored env) and install system deps
RUN set -eux; \
    cat > /etc/apt/sources.list <<'EOF'
deb https://mirrors.aliyun.com/ubuntu/ jammy main restricted universe multiverse
deb https://mirrors.aliyun.com/ubuntu/ jammy-updates main restricted universe multiverse
deb https://mirrors.aliyun.com/ubuntu/ jammy-security main restricted universe multiverse
deb https://mirrors.aliyun.com/ubuntu/ jammy-backports main restricted universe multiverse
EOF

RUN apt-get update -y --fix-missing && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        wget \
        git \
        git-lfs \
        build-essential \
        unzip \
        sox \
        libsox-dev \
        ffmpeg && \
    rm -rf /var/lib/apt/lists/*

RUN git lfs install

# ==================================================================
# conda install and conda forge channel as default
# ------------------------------------------------------------------
# Install miniforge
RUN wget --quiet https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O ~/miniforge.sh && \
    /bin/bash ~/miniforge.sh -b -p /opt/conda && \
    rm ~/miniforge.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo "source /opt/conda/etc/profile.d/conda.sh" >> /opt/nvidia/entrypoint.d/100.conda.sh && \
    echo "source /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate ${VENV}" >> /opt/nvidia/entrypoint.d/110.conda_default_env.sh && \
    echo "conda activate ${VENV}" >> $HOME/.bashrc

ENV PATH=/opt/conda/bin:$PATH

RUN conda config --add channels conda-forge && \
    conda config --set channel_priority strict
# ------------------------------------------------------------------
# ~conda
# ==================================================================

# Create conda environment with python
RUN conda create -y -n ${VENV} python=3.10
ENV CONDA_DEFAULT_ENV=${VENV}
ENV PATH=/opt/conda/bin:/opt/conda/envs/${VENV}/bin:$PATH

WORKDIR /workspace

ENV PYTHONPATH="/workspace/CosyVoice:/workspace/CosyVoice/third_party/Matcha-TTS"

# RUN git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
# Copy local source code instead of cloning from GitHub
COPY . /workspace/CosyVoice

RUN conda activate ${VENV} && conda install -y -c conda-forge pynini==2.1.5
RUN conda activate ${VENV} && cd CosyVoice && \
    pip install -r requirements.txt --no-cache-dir --retries 10 --timeout 120 \
    --index-url https://mirrors.aliyun.com/pypi/simple/ \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/ \
    --trusted-host mirrors.aliyun.com \
    --trusted-host download.pytorch.org \
    --trusted-host aiinfra.pkgs.visualstudio.com

WORKDIR /workspace/CosyVoice

# Expose port for OpenAI compatible API
EXPOSE 8000

# Set default command to run OpenAI compatible API server
ENV MODEL_DIR=iic/CosyVoice-300M
ENV PORT=8000
ENV HOST=0.0.0.0

# Use shell form to expand environment variables
SHELL ["/bin/bash", "--login", "-c"]
CMD python runtime/python/fastapi/openai_compatible.py --model_dir ${MODEL_DIR} --port ${PORT} --host ${HOST}
