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
  start
  help
EOF
}

check_installation() {
  if  [ ! -f ./run.sh ]; then
    die "It seems MELD is not installed. Please install MELD by running meld.sh install"
  fi
}

download_files() {
  if [ ! -f ./run.sh ]; then
    curl -O https://raw.githubusercontent.com/aktin/MELD/refs/heads/main/scripts/run.sh
  fi

  if [ ! -f ./compose.yml ]; then
    curl -O https://raw.githubusercontent.com/aktin/MELD/refs/heads/main/scripts/compose.yml
  fi

  if [ ! -f ./utils.sh ]; then
    curl -O https://raw.githubusercontent.com/aktin/MELD/refs/heads/main/scripts/utils.sh
  fi

  if [ ! -f ./install.sh ]; then
    curl -O https://raw.githubusercontent.com/aktin/MELD/refs/heads/main/scripts/install.sh
  fi
}

print_logo

case "$1" in
  "install")
    shift
    download_files
    "$DIR/install.sh" "$@"
    ;;
  "inference")
    shift
    check_installation
    "$DIR/run.sh" "$@"
    ;;
  "help" | "")
    print_help
    exit 0
    ;;
  *)
    echo "Unexpected argument: $1"
    ;;
esac
