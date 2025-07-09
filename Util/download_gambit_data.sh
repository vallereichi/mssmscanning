#!/bin/bash

show_help() {
    echo "Usage: $0 -f FILE -d DIRECTORY"
    echo
    echo "Options:"
    echo "  -f FILE         Filename of the GAMBIT run to copy"
    echo "  -h              Show this help message and exit"
}
while getopts "f:d:h" opt; do
    case $opt in
        f) FILE=$OPTARG ;;
        h) show_help; exit 0 ;;
        *) echo "Invalid option"; exit 1 ;;
    esac
done
if [ -z "$FILE" ] ; then
    echo "Please specify the filename of the GAMBIT run to copy using -f option."
    show_help
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
scp -r raven:~/GAMBIT/docker/runs/$FILE "$SCRIPT_DIR/../data/gambit-output/"

if [ $? -ne 0 ]; then
    echo "Failed to copy the file. Please check the filename and try again."
    exit 1
fi
