#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
for item in clips landmarks manifest.json curation.json; do
    if [ ! -e "dataset/$item" ]; then
        echo "Missing dataset/$item; restore the prepared dataset first." >&2
        exit 1
    fi
done
mkdir -p tmp
COPYFILE_DISABLE=1 tar -czf tmp/nsl-runtime-data.tar.gz -C dataset clips landmarks manifest.json curation.json
printf '%s\n' 'Ready: tmp/nsl-runtime-data.tar.gz (excluded from Git).'
