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
        """Create database tables if they don't exist and run migrations.

        Startup scenarios handled here:
          * Brand-new install (no DB file): create_all() builds the schema.
          * Normal startup (DB + existing migrations): upgrade() then autogenerate.
          * Restored backup DB with a fresh/empty migrations dir: the DB already
            holds the correct schema and all data, so we treat it as the baseline
            (stamp head) instead of trying to create already-existing tables
            (which would crash with 'table already exists').
          * A previously-disabled module re-enabled: include_object in env.py
            guarantees no data-bearing table is ever dropped by autogeneration.
        """
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

            migrations_empty = not cls._has_migration_scripts(migrations_dir)
            db_has_tables = cls._db_has_data_tables(db_path)

            if migrations_empty and db_has_tables:
                # The DB was created/restored independently of these (now fresh)
                # migrations — e.g. a backup .db was dropped in. The existing tables
                # already match the models (so they won't be dropped/recreated), but
                # any NEW tables introduced since the backup (e.g. files_library)
                # must still be created. So we establish the current schema as the
                # migration baseline: generate a baseline revision from the diff
                # between the models and the DB, then APPLY it (upgrade). The
                # generated baseline only contains the tables/columns that are
                # genuinely missing — existing tables and all their data are left
                # untouched. The include_object guard in env.py additionally
                # guarantees no data-bearing table is ever dropped here.
                print("Migrations dir is empty but DB already has tables — "
                      "establishing current DB schema as the baseline and applying it.")
                # Clear any stale alembic_version row first: a restored backup may
                # carry a revision that references migrations which no longer exist,
                # which would make autogeneration fail with "Can't locate revision".
                cls._clear_alembic_version(db_path)
                flask_migrate(directory=migrations_dir, message='baseline')
                upgrade(directory=migrations_dir)
                print("Database upgraded to baseline. Migration completed successfully.")
                return

            # Apply any existing migrations first (bring DB up to current head)
            print("Applying migrations...")
            upgrade(directory=migrations_dir)

            # Generate a new migration if the models have changed since last revision.
            # This keeps schema updates fully automatic for the user (no manual CLI).
            # Safety: migrations/env.py defines `include_object`, which REFUSES to
            # autogenerate drop_table(...) for data-bearing tables, so a temporarily
            # disabled module can never again trigger a data-wiping migration.
            print("Generating migrations...")
            flask_migrate(directory=migrations_dir)

            # Apply the new migration to update the database schema
            print("Applying new migrations...")
            upgrade(directory=migrations_dir)
            print("Database migration completed successfully.")

    @staticmethod
    def _has_migration_scripts(migrations_dir):
        """True if the versions/ subfolder contains at least one .py migration."""
        versions_dir = os.path.join(migrations_dir, 'versions')
        if not os.path.isdir(versions_dir):
            return False
        return any(name.endswith('.py') for name in os.listdir(versions_dir))

    @staticmethod
    def _db_has_data_tables(db_path):
        """True if the SQLite DB exists and already holds user data tables
        (anything besides the alembic_version bookkeeping table)."""
        if not os.path.exists(db_path):
            return False
        try:
            from sqlalchemy import create_engine, inspect
            engine = create_engine(f"sqlite:///{db_path}")
            try:
                tables = inspect(engine).get_table_names()
            finally:
                engine.dispose()
            return any(t != 'alembic_version' for t in tables)
        except Exception as e:
            print(f"[DatabaseManager] Could not inspect DB tables: {e}")
            return False

    @staticmethod
    def _clear_alembic_version(db_path):
        """Empty the alembic_version bookkeeping table.

        Used when adopting a restored/pre-existing DB as the migration baseline:
        the DB may carry a revision stamp pointing at migrations that no longer
        exist (e.g. after the migrations directory was reset), which would make
        autogeneration fail with 'Can't locate revision'. Clearing it lets the
        baseline path start from a clean slate.
        """
        import sqlite3
        try:
            con = sqlite3.connect(db_path)
            con.execute('DELETE FROM alembic_version')
            con.commit()
            con.close()
        except sqlite3.OperationalError:
            # Table doesn't exist yet — nothing to clear.
            pass

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