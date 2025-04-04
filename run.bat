@echo off
:: Activate the virtual environment
call .env\Scripts\activate

:: Set environment variables for Flask
set FLASK_APP=app.python

set DATA_FOLDER=%DATA_PATH%
:: Check if DATA_FOLDER is set, if not, use default
if "%DATA_FOLDER%"=="" (
    set DATA_FOLDER=project_data
)
:: Run the Python application
python app.py --data-folder "$DATA_FOLDER"