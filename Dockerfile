#-------------------------------------------------------------#
# Simple Dockerfile to build tunnel with local machine in order
# to host Ollama models, if the project is running on a cluster
#-------------------------------------------------------------#

FROM nvidia/cuda:12.2.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive 

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    golang \
    git \
    build-essential \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.11
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.11 python3.11-venv python3.11-dev python3-pip && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --set python3 /usr/bin/python3.11 && \
    rm -rf /var/lib/apt/lists/*

# Verify Python version
RUN python3 --version

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Set working dir
WORKDIR /app

# Copy only requirements first (to enable Docker caching)
COPY requirements.txt .

# Install Python packages early
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Expose and run Ollama
EXPOSE 11434
ENTRYPOINT ["ollama", "serve"]
