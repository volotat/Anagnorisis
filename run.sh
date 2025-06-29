 #!/bin/bash
 
echo "SH: Starting Anagnorisis application..."
echo "SH: Activating virtual environment..."
source "${VENV_PATH:-'.env'}/bin/activate" || echo "Failed to activate virtual environment!"

echo "SH: Setting Flask environment..."
export FLASK_APP=app.py
DATA_FOLDER="${DATA_PATH:-project_data}"

echo "SH: Current working directory: $(pwd)"
echo "SH: Using data folder: $DATA_FOLDER"

echo "SH: Starting Flask application..."
python app.py --data-folder "$DATA_FOLDER"