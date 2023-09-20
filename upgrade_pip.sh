#!/bin/bash

# Upgrade pip to the latest version
pip install --upgrade pip

# Install Python 3.11 using apt-get (for Ubuntu/Debian-based systems)
sudo apt-get update
sudo apt-get install python3.11

# Check if Python 3.11 is available
python3.11 --version

# Make the upgrade_pip.sh script executable
chmod +x upgrade_pip.sh

