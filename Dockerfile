FROM python:3.12-slim

ENV APP_HOME /app
WORKDIR $APP_HOME

COPY advertising_roi_training.py advertising_roi_training.py
COPY test_training.py test_training.py
COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "advertising_roi_training.py"]
