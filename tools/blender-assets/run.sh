#!/usr/bin/env bash
# run.sh — build the game meshes deterministically.
#
#   bash tools/blender-assets/run.sh
#
# Requires Blender 4.x on PATH. Requires numpy inside Blender's Python for the
# glTF exporter; on distro packages that ship without it:
#   python3.12 -m pip install --break-system-packages \
#       --target "$HOME/.config/blender/4.0/scripts/addons/modules" numpy
#
# Threads are pinned to 1 and PYTHONHASHSEED to 0: Blender's object-join and
# modifier internals iterate hash-ordered data whose order depends on the
# per-process Python hash seed, which showed up as run-to-run vertex ORDER
# changes (same coordinates, different sequence -> different sha256).
set -euo pipefail
export PYTHONHASHSEED=0
exec blender -b --threads 1 -P "$(dirname "$0")/build_models.py" "$@"
