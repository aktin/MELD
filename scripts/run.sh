#!/usr/bin/env bash
set -e

source ./utils.sh

# Default values
OUTPUT_DIR="./output"
CONTRACT_FILE=""
QUERY_FILE=""
CMD=""
VERSION="0.1.0-alpha"
IMAGE_NAME="meld-orchestrator"

print_help() {
  cat <<EOF
Usage:
  meld.sh inference [command] [options]

Commands:
  pull                    Pull inference image defined in contract
  run                     Run inference image defined in contract
  delete                  Delete inference image defined in contract

Options:
  -c, --contract           Path of contract, required
  -q, --query              Path of SQL file, overwrites query url in contract
  -o, --output             Output directory, default: ./output
  -h, --help               Show this help message

Examples:
  meld.sh inference run --contract /path/to/query.sql --output ./output
EOF
}

set_env_variables() {
  export MELD_CONTRACT_PATH="$CONTRACT_FILE"
  export MELD_OUTPUT_DIR="$OUTPUT_DIR"
  export MELD_CMD="$CMD"
}

set_docker_env_variables() {
  docker_env=("--env=DB_HOST=host.docker.internal"
              "--env=DB_PORT=5433"
              "--env=DB_USER=i2b2crcdata"
              "--env=DB_PASSWORD=demouser"
              "--env=DB_SCHEMA=i2b2"
              "--env=MELD_CMD=${MELD_CMD}")
}

set_docker_volumes() {
  docker_volumes=("--volume=/var/run/docker.sock:/var/run/docker.sock"
                  "--volume=./logs:/logs"
                  "--volume=./jobs/:/jobs/"
                  "--volume=${MELD_OUTPUT_DIR:-./output}:/output"
                  "--volume=${MELD_CONTRACT_PATH:-./contract.yaml}:/resources/contract.yaml:ro"
                  "--volume=$HOME/.docker:/root/.docker:ro"
                  "--volume=${QUERY_FILE:-./query.sql}:/resources/query.sql:ro")
}

start_container() {
  docker run --add-host host.docker.internal:host-gateway \
      "${docker_env[@]}" \
      "${docker_volumes[@]}" \
      ghcr.io/simhue/${IMAGE_NAME}:${VERSION:-latest} #umbenennen in meld-orchestrator
}

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    "run" | "pull" | "delete")
      CMD="$1"
      shift
      ;;
    -o|--output)
      [[ $# -ge 2 ]] || echo "Missing value for $1"
      OUTPUT_DIR="$2"
      shift 2
      ;;
    -c|--contract)
      [[ $# -ge 2 ]] || echo "Missing value for $1"
      CONTRACT_FILE="$2"
      shift 2
      ;;
    -q|--query)
      [[ $# -ge 2 ]] || echo "Missing value for $1"
      QUERY_FILE="$2"
      shift 2
      ;;
    -h|--help)
      print_help
      exit 0
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "Unknown option: $1"
      break
      ;;
    *)
      echo "Unexpected argument: $1"
      break
      ;;
  esac
done

# Validate required arguments
[[ -n "$CMD" ]] || { echo "Missing command"; exit 1; }
[[ -n "$CONTRACT_FILE" ]] || { echo "Missing required argument: --contract"; exit 1; }

require_file "$CONTRACT_FILE"

if [ -n "$QUERY_FILE" ]; then
  require_file "$QUERY_FILE"
fi

set_env_variables
set_docker_volumes
set_docker_env_variables

start_container
