#!/bin/bash

# Create a virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate the virtual environment
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Install python-dotenv for environment variable management
echo "Installing python-dotenv..."
pip install python-dotenv

echo "Setup complete! Please edit the .env file with your API key and other settings."
echo "Then activate the environment with: source venv/bin/activate"
