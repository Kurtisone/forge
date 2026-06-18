FROM python:3.12

WORKDIR /app

COPY . .

ENV PYTHONPATH=/app/src

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "-m", "forge.main"]
