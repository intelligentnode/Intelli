#!/bin/bash

# Script to install local development version of Intelli in editable mode

echo "Installing local Intelli in editable mode..."

# Navigate to the project root
cd ..

# Uninstall existing version
pip uninstall -y intelli

# Install in editable mode
pip install -e .

echo "✅ Local development version installed!"
echo "You can now run the GPT-5 samples using the local version."
