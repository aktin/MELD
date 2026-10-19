#!/bin/bash
set -euo pipefail

if [ "$(id -u)" = "0" ]; then
    socket_gid=$(stat -c '%g' /var/run/docker.sock)
    if ! getent group "$socket_gid" > /dev/null; then
        groupadd --gid "$socket_gid" "docker-socket-${socket_gid}"
    fi
    usermod --append --groups "$socket_gid" meld
    export HOME=/home/meld
    exec gosu meld "$@"
fi

exec "$@"
