FROM nvcr.io/nvidia/pytorch:19.12-py3 as fairmot-base
RUN apt-get update && apt-get install -y ffmpeg \
  && rm -rf /var/lib/apt/lists/*
COPY ./requirements.txt .
RUN pip install -r requirements.txt


FROM fairmot-base as fairmot
COPY src src
WORKDIR src/lib/models/networks/DCNv2
RUN python setup.py install && rm -rf src
WORKDIR /workspace
