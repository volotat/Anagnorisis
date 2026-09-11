"""
Tests for src/db_models.py — export_db_to_csv() and import_db_from_csv()

Uses an in-memory SQLite database with a minimal ImagesLibrary table so no
files, models, or Docker are needed.

Covers:
  - export produces valid CSV with table.column header format
  - all inserted rows appear in the exported CSV
  - round-trip: export then re-import updates existing rows matched by hash
  - import adds new rows when no hash match exists
  - import skips columns that do not exist in the DB schema
  - excluded_columns are absent from the export
  - datetime columns survive the round-trip
  - empty database exports without crashing
"""
import csv
import io
import pytest
from datetime import datetime
from flask import Flask
from sqlalchemy import Column, Integer, String, Table, text
from src.db_models import db, export_db_to_csv, import_db_from_csv


# ---------------------------------------------------------------------------
# Minimal model for testing
# ---------------------------------------------------------------------------

# We define the test model here rather than importing the real module models
# so this test file has zero dependency on image/music-specific code.

class _TestLibrary(db.Model):
    __tablename__ = 'test_library'
    id         = db.Column(db.Integer, primary_key=True)
    hash       = db.Column(db.String, unique=True, nullable=True)
    file_path  = db.Column(db.String, nullable=True)
    user_rating= db.Column(db.Float,  nullable=True)
    rated_at   = db.Column(db.DateTime, nullable=True)
    embedding  = db.Column(db.LargeBinary, nullable=True)  # should be excluded


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def app():
    """Flask app with an in-memory SQLite DB and the test table."""
    flask_app = Flask(__name__)
    flask_app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
    flask_app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    db.init_app(flask_app)
    with flask_app.app_context():
        # Only this file's table is created. SQLAlchemy metadata is process-wide
        # and src.db_models already registered the application's own tables
        # (files_library), so db.create_all() would put tables in this database
        # that this file does not own — and export_db_to_csv() exports what the
        # database holds, so it would legitimately export them too.
        _TestLibrary.__table__.create(db.engine)
        yield flask_app
    # Drop everything after each test
    with flask_app.app_context():
        _TestLibrary.__table__.drop(db.engine)


@pytest.fixture()
def model_table_without_schema(app):
    """A table that exists as a model but not in the database.

    This is the state the app is in when a module's models are loaded but the
    schema has not been migrated for it yet.
    """
    table = Table(
        'model_only_table', db.Model.metadata,
        Column('id', Integer, primary_key=True),
        Column('hash', String),
    )
    yield table
    with app.app_context():
        db.Model.metadata.remove(table)


@pytest.fixture()
def session(app):
    with app.app_context():
        yield db.session


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _insert(session, **kwargs):
    row = _TestLibrary(**kwargs)
    session.add(row)
    session.commit()
    return row


def _parse_csv(csv_str: str):
    reader = csv.reader(io.StringIO(csv_str), quoting=csv.QUOTE_MINIMAL, escapechar='\\')
    rows = list(reader)
    headers = rows[0] if rows else []
    data = rows[1:]
    return headers, data


def _sections(csv_str: str):
    """Split an export into ``{table: (header_row, data_rows)}``.

    An export holds one section per table and each section has its own header
    row, so the first line of the file is only the first table's header. A row
    starts a section when every cell is ``table.column`` for one table — the
    same rule import_db_from_csv() uses to find a header.
    """
    reader = csv.reader(io.StringIO(csv_str), quoting=csv.QUOTE_MINIMAL, escapechar='\\')
    sections = {}
    current = None
    for row in reader:
        if row and all('.' in cell for cell in row):
            table = row[0].split('.', 1)[0]
            if all(cell.split('.', 1)[0] == table for cell in row):
                current = table
                sections[current] = (row, [])
                continue
        if current is not None:
            sections[current][1].append(row)
    return sections


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestExportDbToCsv:
    def test_export_empty_db(self, session):
        result = export_db_to_csv(session)
        assert isinstance(result, str)

    def test_export_header_format(self, session):
        _insert(session, hash='abc', file_path='a.jpg', user_rating=7.0)
        result = export_db_to_csv(session)
        headers, _ = _parse_csv(result)
        # All header cells must be "table_name.column_name"
        for h in headers:
            assert '.' in h, f"Header '{h}' missing table prefix"
            table, col = h.split('.', 1)
            assert table == 'test_library'

    def test_export_rows_present(self, session):
        _insert(session, hash='h1', file_path='one.jpg', user_rating=8.0)
        _insert(session, hash='h2', file_path='two.jpg', user_rating=5.0)
        result = export_db_to_csv(session)
        _, data = _parse_csv(result)
        assert len(data) == 2

    def test_excluded_columns_absent(self, session):
        _insert(session, hash='hx', file_path='x.jpg', embedding=b'\x00\x01')
        result = export_db_to_csv(session, excluded_columns=['embedding'])
        headers, _ = _parse_csv(result)
        col_names = [h.split('.', 1)[1] for h in headers]
        assert 'embedding' not in col_names

    def test_datetime_exported_as_string(self, session):
        dt = datetime(2024, 6, 15, 12, 30, 0)
        _insert(session, hash='dt1', file_path='d.jpg', rated_at=dt)
        result = export_db_to_csv(session)
        assert '2024-06-15' in result

    def test_export_skips_a_model_table_missing_from_the_database(
            self, session, model_table_without_schema):
        """A model whose table has not been created must not abort the export.

        src/app_factory/database_manager.py reaches this state on a restored
        backup or a re-enabled module: the models are imported, so they are in
        the metadata, while the schema still has to be migrated. Querying the
        missing table used to raise OperationalError and take the whole
        /export_database_csv response down with it.
        """
        _insert(session, hash='abc', file_path='a.jpg', user_rating=7.0)

        result = export_db_to_csv(session)

        sections = _sections(result)
        assert 'model_only_table' not in sections
        assert len(sections['test_library'][1]) == 1

    def test_export_includes_a_database_table_without_a_model(self, session):
        """A table left behind by a disabled or removed module is still exported.

        migrations/env.py deliberately refuses to drop a data-bearing table when
        its module is not loaded, so such a table can exist in the database with
        no model to describe it. It has to survive a backup.
        """
        session.execute(text(
            'CREATE TABLE videos_library (id INTEGER PRIMARY KEY, hash TEXT, user_rating REAL)'
        ))
        session.execute(text(
            "INSERT INTO videos_library (hash, user_rating) VALUES ('v1', 9.5)"
        ))
        session.commit()

        result = export_db_to_csv(session)

        headers, rows = _sections(result)['videos_library']
        assert headers == ['videos_library.id', 'videos_library.hash',
                           'videos_library.user_rating']
        assert rows == [['1', 'v1', '9.5']]

    def test_export_excludes_columns_from_a_database_table_without_a_model(self, session):
        """excluded_columns applies to model-less tables too."""
        session.execute(text(
            'CREATE TABLE videos_library (id INTEGER PRIMARY KEY, hash TEXT, embedding BLOB)'
        ))
        session.commit()

        result = export_db_to_csv(session, excluded_columns=['embedding'])

        headers, _ = _sections(result)['videos_library']
        assert headers == ['videos_library.id', 'videos_library.hash']


class TestImportDbFromCsv:
    def test_import_new_row(self, session):
        # Start with an empty table, import one row
        export = export_db_to_csv(session)  # empty but gets headers
        # Build a minimal CSV for one new row
        csv_data = (
            "test_library.hash,test_library.file_path,test_library.user_rating\r\n"
            "newrow,/new/file.jpg,9.0\r\n"
        )
        import_db_from_csv(session, csv_data)
        rows = session.query(_TestLibrary).all()
        assert len(rows) == 1
        assert rows[0].hash == 'newrow'
        assert rows[0].user_rating == 9.0

    def test_import_updates_existing_row_by_hash(self, session):
        _insert(session, hash='existing_hash', file_path='old.jpg', user_rating=3.0)
        csv_data = (
            "test_library.hash,test_library.file_path,test_library.user_rating\r\n"
            "existing_hash,old.jpg,8.5\r\n"
        )
        import_db_from_csv(session, csv_data)
        row = session.query(_TestLibrary).filter_by(hash='existing_hash').one()
        assert row.user_rating == pytest.approx(8.5)

    def test_import_skips_unknown_columns(self, session):
        csv_data = (
            "test_library.hash,test_library.file_path,test_library.nonexistent_col\r\n"
            "skip_test,/img.jpg,some_value\r\n"
        )
        # Should not raise even though nonexistent_col isn't in the schema
        import_db_from_csv(session, csv_data)
        rows = session.query(_TestLibrary).all()
        assert len(rows) == 1

    def test_round_trip_preserves_data(self, session):
        dt = datetime(2025, 1, 10, 8, 0, 0, 123456)
        _insert(session, hash='rt_hash', file_path='rt.jpg', user_rating=6.5, rated_at=dt)
        csv_str = export_db_to_csv(session, excluded_columns=['embedding'])

        # Clear the table and re-import
        session.query(_TestLibrary).delete()
        session.commit()
        import_db_from_csv(session, csv_str)

        row = session.query(_TestLibrary).filter_by(hash='rt_hash').one()
        assert row.file_path == 'rt.jpg'
        assert row.user_rating == pytest.approx(6.5)
        # Datetime must survive (seconds precision minimum)
        assert row.rated_at.year == 2025
        assert row.rated_at.month == 1
        assert row.rated_at.day == 10

    def test_import_multiple_rows(self, session):
        csv_data = (
            "test_library.hash,test_library.file_path,test_library.user_rating\r\n"
            "h1,/a.jpg,1.0\r\n"
            "h2,/b.jpg,2.0\r\n"
            "h3,/c.jpg,3.0\r\n"
        )
        import_db_from_csv(session, csv_data)
        rows = session.query(_TestLibrary).all()
        assert len(rows) == 3

    def test_import_empty_csv_no_crash(self, session):
        import_db_from_csv(session, "")
        rows = session.query(_TestLibrary).all()
        assert len(rows) == 0


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])

