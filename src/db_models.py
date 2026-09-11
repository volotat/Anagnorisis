from flask_sqlalchemy import SQLAlchemy
import csv
from io import StringIO
from sqlalchemy import inspect, or_, text, MetaData
from sqlalchemy.types import DateTime
from datetime import datetime

# Naming convention so Alembic batch migrations on SQLite can handle
# constraints properly (all constraints get deterministic names).
naming_convention = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

# Initialize SQLAlchemy to make a reference object for main application and extensions to use
db = SQLAlchemy(metadata=MetaData(naming_convention=naming_convention))

# General purpose table for storing information about files and their respective ratings.
class FilesLibrary(db.Model):
  id = db.Column(db.Integer, unique=True, primary_key=True)
  hash = db.Column(db.String, nullable=True, default=None) # place for soft hash derived with blake2 from parts of the file. Could be None if we don't need it.
  hash_algorithm = db.Column(db.String, nullable=True, default=None) # blake2 most of the time or None
  file_path = db.Column(db.String, nullable=True, index=True) # Full url path like this: osfs:///mnt/media/Movies/SomeMovie.mp4
  user_rating = db.Column(db.Float, nullable=True, index=True) # Place for fast lookup that would be synchronized with project_config/memory/ folder
  user_rating_date = db.Column(db.DateTime, nullable=True) # Necessary for training in the future, to know when the user rated the file
  model_rating = db.Column(db.Float, nullable=True, index=True) # Current prediction of the model for this file, based on the all data available to gather (path, metadata, .meta files, etc.)
  model_hash = db.Column(db.String, nullable=True) # Current prediction model hash to know when to update the score

  def as_dict(self):
    return {column.name: getattr(self, column.name) for column in self.__table__.columns}

def export_db_to_csv(db_session, excluded_columns=None):
    """
    Exports all data from all tables in the database to a CSV string,
    excluding specified BLOB columns.

    The tables are read from the database itself, not from the models. The two
    can disagree and both directions matter: a table whose model is loaded while
    the schema has not been migrated yet must not abort the export (querying it
    raised OperationalError and took the whole /export_database_csv response
    with it), and a table that is in the database but has no model right now —
    a module that is disabled or was removed, which the migration guard in
    migrations/env.py deliberately keeps — must not be silently dropped from the
    backup. Columns come from the model when there is one, and from the database
    catalog otherwise.
    """
    if excluded_columns is None:
      excluded_columns = []

    csv_output = StringIO()
    csv_writer = csv.writer(
        csv_output,
        quoting=csv.QUOTE_MINIMAL, 
        escapechar='\\'  
    )

    # alembic_version is bookkeeping, not user data: it was never part of an
    # export before and must not become one now.
    inspector = inspect(db_session.get_bind())
    db_tables = set(inspector.get_table_names()) - {'alembic_version'}
    models = db.Model.metadata.tables

    # Model order first so an unchanged database exports byte-identically to
    # before; then the tables no model claims, in a stable order.
    ordered = [(name, table) for name, table in models.items() if name in db_tables]
    ordered += [(name, None) for name in sorted(db_tables - set(models))]

    for table_name, table in ordered:
        if table is not None:
            column_names = [col.name for col in table.columns if col.name not in excluded_columns]
            rows = [[getattr(row, col) for col in column_names]
                    for row in db_session.query(table).all()]
        else:
            # No model: take the columns from the database catalog and read the
            # rows as plain tuples.
            column_names = [col['name'] for col in inspector.get_columns(table_name)
                            if col['name'] not in excluded_columns]
            quoted_columns = ', '.join('"%s"' % col for col in column_names)
            result = db_session.execute(
                text('SELECT %s FROM "%s"' % (quoted_columns, table_name))
            )
            rows = [tuple(row) for row in result.fetchall()]

        # Write header
        csv_writer.writerow([f'{table_name}.{col}' for col in column_names])

        for row in rows:
            csv_writer.writerow(row)
    return csv_output.getvalue()

def import_db_from_csv(db_session, csv_data):
    """
    Imports data from csv_data (string) into the database, matching by 'hash' or 'file_path'
    where present.
    Only updates columns that appear in CSV and exist in the DB. 
    Skips columns not in the DB, and adds new rows if no match is found.
    """

    # Convert the incoming string to a CSV reader
    reader = csv.reader(StringIO(csv_data), quoting=csv.QUOTE_MINIMAL, escapechar='\\')

    current_table_name = None
    table_columns = []  # Will hold the list of (col_name_in_db, csv_index)
    for row in reader:
        # Detect if this is a header row: all cells should contain "table_name.column"
        # Example: ["music_library.hash", "music_library.file_path", ...]
        if all("." in cell for cell in row) and len(row) > 0:
            # Parse the table name from the first cell
            first_cell = row[0]
            current_table_name = first_cell.split(".", 1)[0]  # e.g. "music_library"
            
            # Collect columns that appear both in DB and CSV
            # Table object from SQLAlchemy metadata
            db_table = db.Model.metadata.tables.get(current_table_name)
            if db_table is None:
                # If table is unknown, skip until next header row
                table_columns = []
                continue
            
            # Build a list of (db_column_name, csv_index)
            # e.g. row = ["music_library.hash", "music_library.file_path", ...]

            # For each CSV column like "music_library.hash", split off after "."
            column_names_from_csv = [col.split(".", 1)[1] for col in row]

            # Filter only existing columns in DB table
            valid_db_cols = set([c.name for c in db_table.columns])
            table_columns = []
            for idx, col_name in enumerate(column_names_from_csv):
                if col_name in valid_db_cols:
                    table_columns.append((col_name, idx))

        else:
            # A data row for the current table
            if not current_table_name or not table_columns:
                # We have data, but no valid table header read yet
                # or table is unknown
                continue

            # Get the correct table from metadata
            db_table = db.Model.metadata.tables[current_table_name]
            # Build a dict of {db_column_name: cell_value} 
            row_data = {}
            for (db_col_name, csv_index) in table_columns:
                if csv_index < len(row):
                    value = row[csv_index]
                    # Convert to datetime if the column is of DateTime type
                    if isinstance(db_table.c[db_col_name].type, DateTime):
                        # Example of the format used in export: 2024-06-29 23:43:21.599813
                        if value:
                            value = datetime.strptime(value, '%Y-%m-%d %H:%M:%S.%f')
                        else:
                            value = None
                    # Convert empty strings to None for non-string fields
                    elif value == "" and not isinstance(db_table.c[db_col_name].type, db.String):
                        value = None
                    row_data[db_col_name] = value

            # Attempt to match an existing record by "hash" or "file_path" if present
            query_filters = []
            if "hash" in row_data and row_data["hash"]:
                query_filters.append(db_table.c.hash == row_data["hash"])
            if "file_path" in row_data and row_data["file_path"]:
                query_filters.append(db_table.c.file_path == row_data["file_path"])

            existing_row = None
            if query_filters:
                existing_row = db_session.query(db_table).filter(
                    or_(*query_filters)
                ).first()

            if existing_row:
                # Update existing row via a proper UPDATE statement.
                # (Querying a Table object directly returns immutable Row objects,
                # so setattr cannot be used here.)
                db_session.execute(
                    db_table.update()
                    .where(or_(*query_filters))
                    .values(**row_data)
                )
            else:
                # Create new row
                insert_dict = {}
                for col, val in row_data.items():
                    insert_dict[col] = val

                # Insert via the ORM
                db_session.execute(db_table.insert().values(**insert_dict))

    # Commit after all rows are processed
    db_session.commit()