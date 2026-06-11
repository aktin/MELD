#!/usr/bin/env bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

print_logo() {
  cat <<EOF
  
  
  
                        ████    ███
    ██                ████   █████                                         ████       ████       ██████████    ████████████████
   ████     ███      ████     ██                                          ██████      █████    ██████████████  ████████████████
  ████    █████      ███                                                  ████████    █████   █████     ██████       ████
  ████  █████       ████    ███    ███    ███      ███          ██████    █████████   █████  █████        █████      ████
  ███  ████         ████    ███    ███  ██████   ██████       █████████   ██████████  █████  █████         ████      ████
  ███████           ████   ████   ████ ███████  ███ ███     █████  ████   █████ █████ █████  ████          ████      ████
  ████████          ███    ████   ███████  ███ ███  ████   ████    ███    █████  ██████████  █████        █████      ████
 ████ ██████        ███    ███    ██████   ██████   ████   ███   █████    █████    ████████  ██████      ██████      ████
 ████    ██████     ███   ████    █████    █████    █████ ████ ████████   █████     ███████   ████████████████       ████
 ████       ██████  ████  ████    ████      ███      █████ ██████  █████  █████      █████      ████████████         ████
 ███           ██   ███    ███     ██                        █       ██     █          ██           ████              ██
 
 
 
EOF
}

print_help() {
  cat <<EOF
Usage:
  $(basename "$0") [command]

Available commands:
  install
  inference
  help
EOF
}

check_installation() {
  if  [ ! -f ./run.sh ]; then
    echo "It seems that MELD is not installed. Please install MELD by running \"./meld.sh install\""
    exit 1
  fi
}

check_docker() {
  if ! command -v docker >/dev/null 2>&1; then
    echo "Docker is not installed or not available in PATH"
    exit 1
  fi

  if docker compose version >/dev/null 2>&1; then
    echo "Docker Compose plugin is installed"
  elif command -v docker-compose >/dev/null 2>&1; then
    echo "Docker Compose standalone is installed"
  else
    echo "Docker Compose is not installed"
    exit 1
  fi
}

download_file_if_not_exist() {
  file_name=$1
  if [ ! -f "./$file_name" ]; then
    curl -o "./$file_name" "https://raw.githubusercontent.com/aktin/MELD/refs/heads/main/scripts/$file_name"
  fi
}

download_files() {
  download_file_if_not_exist run.sh

  download_file_if_not_exist compose.yml

  download_file_if_not_exist utils.sh

  download_file_if_not_exist install.sh
}

print_logo

check_docker

case "$1" in
  "install")
    shift
    download_files
    bash "$DIR/install.sh" "$@"
    ;;
  "inference")
    shift
    check_installation
    bash "$DIR/run.sh" "$@"
    ;;
  "help" | "")
    print_help
    exit 0
    ;;
  *)
    echo "Unexpected argument: $1"
    ;;
esac
