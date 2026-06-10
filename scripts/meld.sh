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
