#!/bin/bash

shoq_help() {
	echo "usage: $0 -s SOURCE -o OUTPUT"
	echo
	echo "Options:"
	echo "	-s SOURCE	source directory where the tarballs are located"
	echo "	-o OUTPUT	output directory"
	echo "	-h help		show this help message"
}
while getopts "s:o:h" opt; do
	case $opt in
		s) SOURCE_DIR=$OPTARG ;;
		o) DEST_DIR=$OPTARG ;;
		h) show_help; exit 0 ;;
		*) echo "invalid option"; exit 1 ;;
	esac
done

# Create the destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

# Loop through all .tar.gz files in the source directory
for archive in "$SOURCE_DIR"/*.tar.gz; do
    # Get the base name of the archive (without path and extension)
    base_name=$(basename "$archive" .tar.gz)

    # # Create a subdirectory for this archive
    target_dir="$DEST_DIR"
    mkdir -p "$target_dir"

    # Extract the archive into its subdirectory
    tar -xzvf "$archive" -C "$target_dir"
done

