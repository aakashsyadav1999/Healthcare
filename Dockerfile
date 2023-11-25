FROM python:3.9.0-buster
COPY . /app
WORKDIR /app
COPY requirements.txt ./
RUN pip3 install -r requirements.txt
CMD python app.py