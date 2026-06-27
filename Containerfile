FROM python:3.12

WORKDIR /app

COPY . .

ENV PYTHONPATH=/app/src

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

# Default: HTTP API (accessible from browser / other machines)
# Override for REPL: podman run -it ... forge-core python -m forge.main
CMD ["uvicorn", "forge.api:app", "--host", "0.0.0.0", "--port", "8000"]
