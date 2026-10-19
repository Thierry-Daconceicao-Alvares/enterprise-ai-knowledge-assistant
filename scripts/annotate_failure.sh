#!/usr/bin/env bash
# Run a command in GitHub Actions; when it fails, show the reason in the run
# summary as an error annotation, so that it can be read without the log.
# Usage: bash scripts/annotate_failure.sh "<title>" <command> [arguments...]
set -uo pipefail

title=$1
shift
log=$(mktemp)
"$@" 2>&1 | tee "$log"
status=${PIPESTATUS[0]}
if [ "$status" -ne 0 ]; then
  # The CLI prints "Error (<code>): <message>"; Python exceptions end with "<Name>Error: <message>".
  reason=$(grep -E '^(Error \(|[A-Za-z]*Error: )' "$log" | tail -n 1)
  if [ -z "$reason" ]; then
    reason=$(grep -v '^[[:space:]]*$' "$log" | tail -n 1)
  fi
  reason=${reason//'%'/'%25'}
  echo "::error title=${title}::${reason:-exit code ${status}}"
fi
rm -f "$log"
exit "$status"
