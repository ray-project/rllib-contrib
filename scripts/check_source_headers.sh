#!/bin/bash
HEADER="# Copyright 2023-onwards Anyscale, Inc. The use of this library is subject to the
# included LICENSE file."

BOLD_WHITE='\033[1;37m'

status=0
for path in "$@"
do
    text=`cat $path`
    if ! [[ "$text" =~ ^$HEADER ]]; then
        printf "${BOLD_WHITE}added license to $path"
        echo "$(echo "$HEADER" | cat - $path)" > $path
        status=1
    fi
done

exit $status
