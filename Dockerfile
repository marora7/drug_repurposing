FROM nvidia/cuda:11.7.1-base-ubuntu20.04

RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common \
    libsm6 libxext6 libxrender-dev curl \
    && rm -rf /var/lib/apt/lists/*

RUN echo "**** Installing Python ****" && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.7 \
    python3-pip \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*
RUN apt update && apt upgrade -y
RUN apt-get autoclean -y \
    && apt-get autoremove -y

# Set app directory as working directory
WORKDIR /app

COPY ./requirements.txt /app
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy all files to app directory
COPY . /app

CMD python3 generate_data.py
#ENTRYPOINT ["tail", "-f", "/dev/null"]
