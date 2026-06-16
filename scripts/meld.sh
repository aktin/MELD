#!/usr/bin/env bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

scripts=("run.sh"
          "utils.sh"
          "install.sh"
          "compose.yml")

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
  update
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

#  if docker compose version >/dev/null 2>&1; then
#    echo "Docker Compose plugin is installed"
#  elif command -v docker-compose >/dev/null 2>&1; then
#    echo "Docker Compose standalone is installed"
#  else
#    echo "Docker Compose is not installed"
#    exit 1
#  fi
}

download_file_if_not_exist() {
  file_name=$1
  if [ ! -f "./$file_name" ]; then
    curl -o "./$file_name" "https://raw.githubusercontent.com/aktin/MELD/refs/heads/main/scripts/$file_name"
  fi
}

download_scripts() {
  for s in "${scripts[@]}"; do
    download_file_if_not_exist "$s"
  done
}

delete_scripts() {
  for s in "${scripts[@]}"; do
    rm "./$s"
  done
}

print_logo

check_docker


case "$1" in
  "install")
    download_scripts
    bash "$DIR/install.sh" "$@"
    ;;
  "update")
    delete_scripts
    download_scripts
    bash "$DIR/install.sh" "$@"
    ;;
  "uninstall")
    bash "$DIR/install.sh" "$@"
    delete_scripts
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
