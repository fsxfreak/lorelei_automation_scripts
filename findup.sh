#!/usr/bin/env bash
# https://unix.stackexchange.com/questions/6463/find-searching-in-parent-directories-instead-of-subdirectories
set -e
path="$1"
shift 1
while [[ $path != / ]];
do
    find "$path" -maxdepth 1 -mindepth 1 "$@"
    # Note: if you want to ignore symlinks, use "$(realpath -s "$path"/..)"
    path="$(readlink -f "$path"/..)"
done
