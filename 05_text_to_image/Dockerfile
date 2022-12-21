FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime

ENV PYTHONUNBUFFERED=1
ARG WORKDIR=/cv_lab
WORKDIR ${WORKDIR}

RUN apt-get update && \
    apt-get install -y git gcc g++ wget ffmpeg libsm6 libxrender-dev libxext6 rustc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
RUN pip install --no-warn-script-location --no-cache-dir -r requirements.txt && rm requirements.txt

ENTRYPOINT ["/bin/sh", "-c"]
