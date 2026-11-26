#!/usr/bin/env bash
# Run a command from this repository's root with OpenFOAM (if installed) and the Python environment loaded.
#   bash scripts/run.sh python -m turboai_damage.<module> ...
# Environment variables: TURBOAI_VENV (default ~/venv-turboai), TURBOAI_RUNS (default ~/turboai-runs).
args=("$@")
set --   # OpenFOAM's bashrc parses the positional parameters
OF_BASHRC=$(ls -d /usr/lib/openfoam/openfoam*/etc/bashrc 2>/dev/null | sort -V | tail -1)
# shellcheck disable=SC1090
[ -n "$OF_BASHRC" ] && source "$OF_BASHRC"
# shellcheck disable=SC1091
source "${TURBOAI_VENV:-$HOME/venv-turboai}/bin/activate"
export TURBOAI_RUNS=${TURBOAI_RUNS:-$HOME/turboai-runs}
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
exec "${args[@]}"
