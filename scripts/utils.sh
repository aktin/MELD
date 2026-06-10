#!/usr/bin/env bash

DOCKER_COMPOSE_CMD=()

die() {
  echo "Error: $*" >&2
  exit 1
}

script_dir() {
  cd -- "$(dirname -- "${BASH_SOURCE[1]}")" && pwd
}

get_docker_compose_cmd() {
  if docker compose version >/dev/null 2>&1; then
    DOCKER_COMPOSE_CMD=(docker compose)
  elif command -v docker-compose >/dev/null 2>&1; then
    DOCKER_COMPOSE_CMD=(docker-compose)
    echo "docker compose not found, falling back to docker-compose." >&2
  else
    die "Neither 'docker compose' nor 'docker-compose' was found. Please install Docker Compose."
  fi
}

require_directory() {
  local folder="$1"
  local description="${2:-Directory}"

  [[ -n "$folder" ]] || die "$description path is empty"
  [[ -e "$folder" ]] || die "$description '$folder' does not exist"
  [[ -d "$folder" ]] || die "$description '$folder' is not a directory"
}

require_file() {
  local file="$1"
  local description="${2:-File}"

  [[ -n "$file" ]] || die "$description path is empty"
  [[ -e "$file" ]] || die "$description '$file' does not exist"
  [[ -f "$file" ]] || die "$description '$file' is not a regular file"
}