#!/bin/bash
# Script para ejecutar pygame_test.py con el PYTHONPATH correcto

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH="$SCRIPT_DIR/src:$PYTHONPATH"

cd "$SCRIPT_DIR"
python src/game/pygame_test.py "$@"

