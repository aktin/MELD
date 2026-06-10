#!/usr/bin/env bash
set -e

source ./utils.sh

# Default values
OUTPUT_DIR="./output"
COMPOSE_FILE="./compose.yml"
CONTRACT_FILE=""
QUERY_FILE=""
CMD=""

print_help() {
  cat <<EOF
Usage:
  meld.sh start [options]

Options:
  -r, --resources path     Path to resources for contract and query file, required
  -c, --contract           Path or URL of contract, required
  -q, --query              Path of SQL file
  -o, --output DIR         Output directory, default: ./output
  -f, --file               Custom docker compose file
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

start_container() {
  "${DOCKER_COMPOSE_CMD[@]}" -f "$COMPOSE_FILE" up
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
    -f|--file)
      [[ $# -ge 2 ]] || echo "Missing value for $1"
      COMPOSE_FILE="$2"
      shift 2
      ;;
    -c|--contract)
      [[ $# -ge 2 ]] || echo "Missing value for $1"
      CONTRACT_FILE="$2"
      shift 2
      ;;
    -q|--query)
      [[ $# -ge 2 ]] || echo "Missing value for $1"
      echo "Not used at the moment"
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

get_docker_compose_cmd

require_file "$CONTRACT_FILE"

set_env_variables

start_container
