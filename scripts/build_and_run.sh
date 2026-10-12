/bin/sh /home/shuening/code/KlimaNot/scripts/build.sh
#docker push ghcr.io/simhue/meld-orchestrator:latest
docker push ghcr.io/simhue/meld-orchestrator:0.2.1-SNAPSHOT

#export DOCKER_SOCKET_GID=$(stat -c '%g' /var/run/docker.sock)
docker compose up -d
