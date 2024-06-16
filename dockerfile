FROM python:3.12

WORKDIR /src
EXPOSE 80
EXPOSE 443

COPY ./script .

RUN pip install --no-cache-dir -r requirements.txt
RUN python ./app.py

ENV PYTHONUNBUFFERED=1

CMD ["python", "app.py"]