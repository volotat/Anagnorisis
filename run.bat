@echo off
:: Activate the virtual environment
call .env\Scripts\activate

:: Run database migrations
flask db migrate
:: Upgrade the database
flask db upgrade

:: Run the Python application
python app.py