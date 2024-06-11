source .env/bin/activate
export FLASK_APP="app:init_app"
flask db migrate
flask db upgrade
python3 app.py
