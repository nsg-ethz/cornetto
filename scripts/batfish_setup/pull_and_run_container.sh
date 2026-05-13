#!/usr/bin/env bash
set -e

# Free ports used by Batfish/Jupyter
fuser -k 8888/tcp || true
fuser -k 9996/tcp || true

# Stop/remove any existing batfish container cleanly
if docker ps -a --format '{{.Names}}' | grep -q '^batfish$'; then
	docker stop batfish || true
	docker rm batfish || true
fi

# Ensure the data volume exists
docker volume create batfish-data >/dev/null

# Pull latest image and start detached to avoid blocking on Jupyter logs
docker pull iprotogeros/batfish-allione:forwarding-analysis-0.1
docker run -d --name batfish -v batfish-data:/data -p 8888:8888 -p 9996:9996 iprotogeros/batfish-allione:forwarding-analysis-0.1