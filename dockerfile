FROM python:3.12

WORKDIR /src
EXPOSE 80
EXPOSE 443
EXPOSE 7860

COPY ./script .

RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONUNBUFFERED=1

CMD ["python", "train.py"]
