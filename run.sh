export FLASK_APP="app:init_app"
flask db migrate
flask db upgrade
source .env/bin/activate
python3 app.py