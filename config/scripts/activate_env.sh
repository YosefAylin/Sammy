#!/bin/bash
# Activation script for Sammy AI virtual environment
# This is a wrapper that calls the actual script in config/

exec "$(dirname "$0")/config/scripts/activate_env.sh" "$@"