"""The Anagnorisis engine.

Everything here works without a web server, a database or a browser: describing
a file, embedding it, and searching what has been indexed. Three things use it —
the Flask application, the `anagnorisis` command line, and the data server —
and they all call the same functions.

Two rules keep it that way, and both are enforced rather than documented:

* **Nothing app-shaped.** No Flask, no SQLAlchemy, no socketio. Ratings and play
  counts live in the application's database and are passed in; this package
  persists nothing except its own caches. See `tests/test_core_boundary.py`.
* **Progress is a two-method protocol.** Long operations take an object with
  `check()` and `update(progress, message)`. The app passes its task context so
  work is pausable from the Task Manager; the CLI passes something that prints.

Start at `api.py` — it is the whole surface. `cli.py` is argparse over it, so the
command line can never drift from what the app does.
"""
