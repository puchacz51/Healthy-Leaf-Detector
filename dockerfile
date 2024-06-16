FROM python:3.12

WORKDIR /src

COPY ./script .

RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONUNBUFFERED=1

CMD ["python", "app.py"]