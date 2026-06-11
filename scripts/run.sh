#!/usr/bin/env bash
set -e

source ./utils.sh

# Default values
OUTPUT_DIR="./output"
CONTRACT_FILE=""
QUERY_FILE=""
CMD=""
VERSION="0.1.0"

print_help() {
  cat <<EOF
Usage:
  meld.sh start [options]

Options:
  -c, --contract           Path or URL of contract, required
  -q, --query              Path of SQL file
  -o, --output DIR         Output directory, default: ./output
  -v, --version            Define MELD version
  -h, --help               Show this help message

Examples:
  meld.sh install --resources /path/to/resources/ --output ./output
EOF
}

set_env_variables() {
  export MELD_CONTRACT_PATH="$CONTRACT_FILE"
  export MELD_OUTPUT_DIR="$OUTPUT_DIR"
  export MELD_CMD="$CMD"
}

set_docker_env_variables() {
  docker_env=("-e DB_HOST=host.docker.internal"
              "-e DB_PORT=5433"
              "-e DB_USER=i2b2crcdata"
              "-e DB_PASSWORD=demouser"
              "-e DB_SCHEMA=i2b2"
              "-e MELD_CMD=${MELD_CMD}")

}

set_docker_volumes() {
  docker_volumes=("-v /var/run/docker.sock:/var/run/docker.sock"
                  "-v ./logs:/logs"
                  "-v ./jobs/:/jobs/"
                  "-v ${MELD_OUTPUT_DIR:-./output}:/output"
                  "-v ${MELD_CONTRACT_PATH:-./}:/resources/contract.yaml:ro"
                  "-v $HOME/.docker:/root/.docker:ro")

  if [ -n "$QUERY_FILE" ]; then
    docker_env+=("-v $QUERY_FILE:/resources/query.sql:ro")
  fi
}

start_container() {
  docker run --add-host host.docker.internal:host-gateway \
    "${docker_env[@]}" \
    "${docker_volumes[@]}" \
    ghcr.io/simhue/meld:${VERSION:-latest}
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

start_container
