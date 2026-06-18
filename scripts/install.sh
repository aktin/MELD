#!/usr/bin/env bash
set -e

source ./utils.sh

DOCKER_IMAGE="ghcr.io/simhue/meld-orchestrator"
VERSION="0.1.1-alpha"

print_help() {
  cat <<EOF
Usage:
  $(basename "$0") [options]

Options:
  -h, --help             Show this help message

EOF
}

pull_docker_image() {
  docker compose pull
}

update_docker_image() {
  docker compose pull
}

remove_docker_image() {
  docker ps -a --filter "ancestor=$DOCKER_IMAGE:$VERSION" --format "{{.ID}}" | xargs -r docker rm -f
  docker ps -a --filter "ancestor=$DOCKER_IMAGE:latest" --format "{{.ID}}" | xargs -r docker rm -f

  docker rmi "$DOCKER_IMAGE:$VERSION" 2>/dev/null || true
  docker rmi "$DOCKER_IMAGE:latest" 2>/dev/null || true
}

for arg in "$@"; do
  case "$arg" in
    -h | --help)
      print_help
      exit 0
      ;;
  esac
done

case "$1" in
  install)
    pull_docker_image
    ;;
  update)
    update_docker_image
    ;;
  uninstall)
    remove_docker_image
    ;;
esac