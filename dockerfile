# Use NVIDIA's CUDA base image
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

WORKDIR /src

EXPOSE 80
EXPOSE 443
EXPOSE 7860

COPY ./script .

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

ENV PYTHONUNBUFFERED=1

CMD ["python3", "train.py"]
