#!/bin/bash

# 1. Navigate to the folder where this script is located 
cd "$(dirname "$0")" || exit

echo ""
echo "  Starting Calgary Townhouse ROI Predictor..."
echo ""
echo ""

# 2. Check if the virtual environment exists, ow create it
if [ ! -d ".venv" ]; then
    echo "First time setup detected. Creating virtual environment..."
    python3 -m venv .venv
    
    source .venv/bin/activate
    
    pip install -r requirements.txt
    
    echo "Installation complete!"
    echo ""
else
    echo "Virtual environment found. Activating..."
    source .venv/bin/activate
fi

# 3. Change into the Website folder
cd Website || exit

# 4. Automatically open the web browser (Handles both Mac and Linux)
echo "Opening browser..."
if command -v open > /dev/null; then
    open http://127.0.0.1:8000/       # Mac command
elif command -v xdg-open > /dev/null; then
    xdg-open http://127.0.0.1:8000/   # Linux command
fi

# 5. Start the Flask server
echo "Starting Python server... (Do not close this window)"
echo ""
python3 interface.py

# 6. Move back up to the main folder when the app shuts down
cd ..