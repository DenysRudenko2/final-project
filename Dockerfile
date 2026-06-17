FROM python:3.12-slim

WORKDIR /srv

# Залежності
COPY app/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Код + скрипт тренування
COPY model/ ./model/
COPY app/ ./app/

# Тренуємо модель під час білду → model/model.pkl запікається в образ
RUN python model/train.py

ENV MODEL_PATH=/srv/model/model.pkl
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
