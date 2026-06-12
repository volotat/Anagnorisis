import os
import io
from flask import send_file
from flask_migrate import Migrate, upgrade, migrate as flask_migrate, init as flask_init
from src.db_models import db, export_db_to_csv, import_db_from_csv

class DatabaseManager:
    """Manages SQLAlchemy binding, Flask-Migrate, and Database IO Operations."""

    @classmethod
    def init_app(cls, app):
        app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{app.paths['database']}"
        db.init_app(app)
        # Store migrate instance on app if needed later
        app.migrate = Migrate(app, db, directory=app.paths['migrations'], render_as_batch=True)
        
        cls._register_csv_routes(app)

    @classmethod
    def create_and_migrate(cls, app):
        """Create database tables if they don't exist and run migrations."""
        db_path = app.paths['database']
        
        if not os.path.exists(db_path):
            print(f"Database not found at {db_path}. Creating database...")
            with app.app_context():
                db.create_all()
            print("Database created successfully.")
        else:
            print(f"Database found at {db_path}.")

        migrations_dir = app.paths['migrations']
        print(f"Using migrations directory: {migrations_dir}")
        app.migrate.directory = migrations_dir
        
        with app.app_context():
            # Check if migrations directory exists
            if not os.path.exists(migrations_dir):
                print("Initializing migrations...")
                flask_init(directory=migrations_dir)

            # Apply any existing migrations first (bring DB up to current head)
            print("Applying migrations...")
            upgrade(directory=migrations_dir)

            # Generate a new migration if the models have changed since last revision
            print("Generating migrations...")
            flask_migrate(directory=migrations_dir)

            # Apply the new migration to update the database schema
            print("Applying new migrations...")
            upgrade(directory=migrations_dir)
            print("Database migration completed successfully.")  

    @classmethod
    def _register_csv_routes(cls, app):
        @app.route('/export_database_csv')
        @app.auth_decorator
        def export_database_csv():
            """
            Exports all database tables to a CSV file, excluding BLOB data, and sends it as a response.
            """
            # Exclude embedding column
            csv_data = export_db_to_csv(db.session, excluded_columns=['embedding', 'chunk_embeddings'])  
            
            csv_file = io.BytesIO()
            csv_file.write(csv_data.encode('utf-8'))
            csv_file.seek(0)

            return send_file(
                csv_file,
                mimetype='text/csv',
                as_attachment=True,
                download_name='database_export.csv'
            )