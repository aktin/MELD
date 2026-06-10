#!/usr/bin/env bash
set -e

source ./utils.sh

INSTALL_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DOCKER_COMPOSE_FILE="./compose.yml"

print_help() {
  cat <<EOF
Usage:
  $(basename "$0") [options]

Options:
  -f, --file PATH        Docker compose file, default: ./compose.yml
  -h, --help             Show this help message

EOF
}

pull_docker_image() {
  "${DOCKER_COMPOSE_CMD[@]}" -f "$DOCKER_COMPOSE_FILE" pull
}

case "$1" in
  "-f" | "--file")
    shift
    DOCKER_COMPOSE_FILE="$1"
    ;;
  "-h" | "--help")
    print_help
    exit 0
    ;;
esac

get_docker_compose_cmd

pull_docker_image
