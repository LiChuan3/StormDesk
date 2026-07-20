#!/usr/bin/env bash
# Sync the local stormdesk code to the servers (code lives locally, runs remotely).
# Usage: bash server/sync.sh [pro6000-1] [pro6000-3] [pro6000-4]
set -e
cd "$(dirname "$0")/.."
NODES=("$@")
if [ ${#NODES[@]} -eq 0 ]; then NODES=(pro6000-1 pro6000-3); fi
for n in "${NODES[@]}"; do
  case $n in
    pro6000-1|pro6000-2) DEST=/data/yuxiaoning/projects/stormdesk ;;
    pro6000-3|pro6000-4) DEST=/data_hdd/yuxiaoning/projects/stormdesk ;;
    *) echo "unknown node $n"; exit 1 ;;
  esac
  echo "sync -> $n:$DEST"
  tar --exclude=.git --exclude=__pycache__ --exclude='*.pyc' -czf - . \
    | ssh "$n" "mkdir -p $DEST && tar -xzf - -C $DEST"
done
echo "sync complete"
