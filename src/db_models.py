from flask_sqlalchemy import SQLAlchemy
import csv
from io import StringIO
from sqlalchemy import inspect, or_
from sqlalchemy.types import DateTime
from datetime import datetime

# Initialize SQLAlchemy to make a reference object for main application and extensions to use
db = SQLAlchemy()

def export_db_to_csv(db_session, excluded_columns=None):
    """
    Exports all data from all tables in the database to a CSV string,
    excluding specified BLOB columns.
    """
    if excluded_columns is None:
      excluded_columns = []

    csv_output = StringIO()
    csv_writer = csv.writer(
        csv_output,
        quoting=csv.QUOTE_MINIMAL, 
        escapechar='\\'  
    )

    for table_name, table in db.Model.metadata.tables.items():
        # Get column names from the table
        inspector = inspect(db.engine)
        column_names = [col.name for col in table.columns if col.name not in excluded_columns ]

        # Write header
        csv_writer.writerow([f'{table_name}.{col}' for col in column_names])

        # Fetch data for the current table
        query = db_session.query(table)
        for row in query.all():
                row_data = []
                for col in column_names:
                    row_data.append(getattr(row, col))
                csv_writer.writerow(row_data)
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
                # Update existing row
                for col_name, value in row_data.items():
                    setattr(existing_row, col_name, value)
            else:
                # Create new row
                insert_dict = {}
                for col, val in row_data.items():
                    insert_dict[col] = val

                # Insert via the ORM
                db_session.execute(db_table.insert().values(**insert_dict))

    # Commit after all rows are processed
    db_session.commit()