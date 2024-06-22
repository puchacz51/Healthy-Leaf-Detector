# Use NVIDIA's CUDA base image
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

WORKDIR /src



COPY ../script .

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip

RUN pip3 install --no-cache-dir -r requirements.txt

ENV PYTHONUNBUFFERED=1

CMD ["python3", "train.py"]
