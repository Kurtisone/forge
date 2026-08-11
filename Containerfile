FROM python:3.12

WORKDIR /app

COPY . .

ENV PYTHONPATH=/app/src

RUN pip install --no-cache-dir -r requirements.txt

# journalctl (from the systemd package -- reads journal files
# directly, no daemon involved) and the podman CLI client, needed by
# graphs/sysadmin.py. Neither runs as a service inside this
# container; both are just client binaries talking to whatever's
# mounted/proxied in -- see deploy/README.md for the read-only
# access design (bind-mounted journal dir, filtered D-Bus proxy,
# read-only podman API proxy).
RUN apt-get update && apt-get install -y --no-install-recommends \
    systemd \
    podman \
    && rm -rf /var/lib/apt/lists/*

EXPOSE 8000

# Default: HTTP API (accessible from browser / other machines)
# Override for REPL: podman run -it ... forge-core python -m forge.main
CMD ["uvicorn", "forge.api:app", "--host", "0.0.0.0", "--port", "8000"]
