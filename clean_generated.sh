#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

confirm=false
dryrun=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --confirm) confirm=true ; shift ;;
    --dry-run) dryrun=true ; shift ;;
    *) echo "Unknown option: $1" >&2 ; exit 1 ;;
  esac
done

mapfile -t targets < <(find data data/test -maxdepth 1 -type f \( -name 'generated*' -o -name 'sample_inputs.csv' \) 2>/dev/null)

if [[ ${#targets[@]} -eq 0 ]]; then
  echo "No generated files found."
  exit 0
fi

echo "Targets:"
for t in "${targets[@]}"; do
  echo "  $t"

done

if [[ "$dryrun" == true && "$confirm" == false ]]; then
  echo "(dry run only)"
  exit 0
fi

if [[ "$confirm" != true ]]; then
  echo "Pass --confirm to delete these files." >&2
  exit 1
fi

for t in "${targets[@]}"; do
  if rm "$t"; then
    echo "Deleted $t"
  else
    echo "Failed to delete $t" >&2
  fi
done
