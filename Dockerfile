FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime

ENV FORCE_CUDA=1
ARG WORKDIR=/cv_labs
WORKDIR ${WORKDIR}

RUN apt-get update && \
    apt-get install -y git gcc g++ wget ffmpeg libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install MMCV
RUN pip install --no-cache-dir --upgrade pip wheel setuptools && \
    pip install -U mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12.0/index.html

# Install MMDetection
RUN conda clean --all && \
    git clone https://github.com/open-mmlab/mmdetection.git ${WORKDIR} && \
    pip install --no-cache-dir -r requirements/build.txt && \
    pip install --no-cache-dir -e .

# RUN wget https://download.openmmlab.com/mmdetection/v2.0/detectors/detectors_htc_r50_1x_coco/detectors_htc_r50_1x_coco-329b1453.pth
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["/bin/sh", "-c"]
