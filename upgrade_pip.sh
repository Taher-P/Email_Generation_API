# Update pip
pip install --upgrade pip

# Install Python 3.11 using pyenv
pyenv install 3.11.0

# Set Python 3.11 as the global version
pyenv global 3.11.0

# Make sure pip is using the correct Python version
pip install --upgrade pip

# Make the upgrade_pip.sh script executable
chmod +x upgrade_pip.sh

