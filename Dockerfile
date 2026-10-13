FROM python:3.12-slim

ENV APP_HOME /app
WORKDIR $APP_HOME

COPY advertising_model_training.py advertising_model_training.py
COPY test_python.py test_python.py
COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "advertising_model_training.py"]
