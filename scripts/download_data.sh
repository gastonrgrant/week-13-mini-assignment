#!/usr/bin/env bash
# Download NYC TLC high-volume for-hire trips for Q1 2023 and the zone lookup.
# Source: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW="$ROOT/data/raw"
mkdir -p "$RAW"

BASE="https://d37ci6vzurychx.cloudfront.net"
files=(
  "trip-data/fhvhv_tripdata_2023-01.parquet"
  "trip-data/fhvhv_tripdata_2023-02.parquet"
  "trip-data/fhvhv_tripdata_2023-03.parquet"
  "misc/taxi_zone_lookup.csv"
)

for rel in "${files[@]}"; do
  name="$(basename "$rel")"
  dest="$RAW/$name"
  if [[ -f "$dest" ]]; then
    echo "already present: $name"
    continue
  fi
  echo "downloading $name"
  curl -L --fail --retry 3 --continue-at - -o "$dest" "$BASE/$rel"
done

echo
echo "On-disk size:"
du -sh "$RAW"
du -h "$RAW"/*
