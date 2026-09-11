# Creating an Anagnorisis Module

A **module** is a self-contained folder inside `modules/` that adds a new media type or functionality to the application. Modules are auto-discovered at startup — no changes to the core app needed.

> **Install by cloning:** Modules will be installable by running `git clone <repo-url>` inside `modules/` and restarting. Keep **all** module files inside your folder.

**See also:**
- [ARCHITECTURE.md](ARCHITECTURE.md) — How auto-discovery works, file responsibilities, shared utilities reference
- [BEST_PRACTICES.md](BEST_PRACTICES.md) — Code style, naming conventions, common pitfalls, and development tips

---

## Quick start

```bash
# 1. Copy the template
cp -r modules/_module_template modules/my_module

# 2. Rename all occurrences of "example" / "Example" to your module name
#    (in serve.py, db_models.py, page.html, js/main.js)

# 3. Add a config section in config.yaml
# 4. Restart the application — done!
```

---

## Module structure

```
modules/my_module/
├── serve.py               # REQUIRED — entry point, socket events, Flask routes
├── page.html              # REQUIRED — frontend HTML (injected into base.html)
├── js/
│   └── main.js            # REQUIRED — frontend JavaScript (ES module)
├── db_models.py           # OPTIONAL — persistent storage (ratings, play counts, …)
├── train.py               # OPTIONAL — universal evaluator training hook
├── config.defaults.yaml   # OPTIONAL — module config defaults (auto-merged at startup)
├── requirements.txt       # OPTIONAL — Python dependencies (pip install -r)
├── CHANGELOG.md           # OPTIONAL — version history (for downloadable modules)
└── README.md              # OPTIONAL — module-specific documentation
```

---

## Step-by-step guide

### 1. Declare a media type — there is no engine to write

Modules used to subclass `BaseSearchEngine` and wrap their own embedding model. They no longer do: one multimodal model embeds every kind of content, and one `ContentSearch` engine serves all of them.

Your module declares which **media type** it shows, in `config.defaults.yaml`:

```yaml
my_module:
  media_directory: osfs:///mnt/media/my_module/
  media_types: [images]     # a type defined in media_types/media_types.yaml
```

and asks for the shared engine, scoped to that type:

```python
from anagnorisis_core.search.content_search import get_content_search

search_engine = get_content_search(cfg, 'images')
search_engine.initiate(models_folder=cfg.main.embedding_models_path)
```

That gives you `process_files()`, `process_text()`, `compare()`, `get_metadata()`, `get_file_hash()` and the embedding cache, with no per-module code.

If your module handles a kind of content no existing type covers, add it to [`anagnorisis_core/data/media_types/media_types.yaml`](../../anagnorisis_core/data/media_types/media_types.yaml) — extensions, which internal-metadata reader to use, and optionally a tag vocabulary in `anagnorisis_core/data/media_types/tags/`. If it also needs a *new* way of extracting internal metadata, add a reader to `anagnorisis_core/extractors/`.

The base class provides: model downloading, two-level caching, hash computation, and batch processing with progress callbacks.

### 2. Create the database model (`db_models.py`) — *optional*

Only create this file if your module stores per-file data (ratings, play counts, etc.). If your module doesn't need persistent storage, skip it entirely — the app handles its absence gracefully.

When you do need it:

```python
from src.db_models import db

class MyModuleLibrary(db.Model):
    id = db.Column(db.Integer, unique=True, primary_key=True)
    hash = db.Column(db.String, nullable=True, unique=True)
    hash_algorithm = db.Column(db.String, nullable=True, default=None)
    file_path = db.Column(db.String, nullable=True)
    user_rating = db.Column(db.Float, nullable=True)
    user_rating_date = db.Column(db.DateTime, nullable=True)
    model_rating = db.Column(db.Float, nullable=True)
    model_hash = db.Column(db.String, nullable=True)
    # Add your own columns here

    def as_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}
```

Keep the standard columns (`hash`, `user_rating`, `model_rating`, etc.) for compatibility with `FileManager`, `CommonFilters`, and CSV export/import.

### 3. Wire everything up in `serve.py`

The `init_socket_events` function is your module's `main()`:

```python
def init_socket_events(socketio, app=None, cfg=None, data_folder='./project_data'):
    # 1. CommonSocketEvents for status reporting
    common_socket_events = CommonSocketEvents(socketio, module_name="my_module")

    # 2. Read config
    media_directory = cfg.my_module.media_directory

    # 3. Init search engine
    engine = MyModuleSearch(cfg=cfg)
    engine.initiate(models_folder=cfg.main.embedding_models_path,
                    cache_folder=cfg.main.cache_path)

    # 4. Init FileManager (only if you have db_models.py)
    # 5. Init MetadataSearch + CommonFilters
    # 6. Register Flask routes (@app.route)
    # 7. Register SocketIO handlers (@socketio.on)

    common_socket_events.show_loading_status('Initialization complete')
```

### 4. Build the frontend (`page.html` + `js/main.js`)

- `page.html` is an HTML fragment — no `<html>`/`<head>`/`<body>` tags needed.
- Load your JS as `<script type="module" src="modules/my_module/js/main.js"></script>`.
- The global `socket` variable is already available from `base.html`.
- Import shared components:

```javascript
import SearchBarComponent   from '/modules/SearchBarComponent.js';
import FileGridComponent    from '/modules/FileGridComponent.js';
import PaginationComponent  from '/modules/PaginationComponent.js';
import StarRatingComponent  from '/modules/StarRating.js';
import ContextMenuComponent from '/modules/ContextMenuComponent.js';
import MetaEditor           from '/modules/MetaEditor.js';
```

### 5. Add configuration

Create a `config.defaults.yaml` in your module folder with sensible defaults. This is auto-merged at startup — users can override any value in the root `config.yaml`:

```yaml
# modules/my_module/config.defaults.yaml
my_module:
  media_directory: null          # null = disabled at startup
  media_formats:
    - .ext1
    - .ext2
  embedding_model: "namespace/model-name"
```

Users can then override in the root `config.yaml`:

```yaml
my_module:
  media_directory: /path/to/media/files
```

### 6. Integrate with the universal evaluator (`train.py`) — *optional*

Anagnorisis does **not** use per-module evaluator models. Instead, a single **universal evaluator** is trained on user-rated items from **all** modules. The training system auto-discovers modules by globbing `modules/*/train.py` and calling `get_training_pairs()`.

To integrate your module, create `train.py` that exposes this function:

```python
def get_training_pairs(cfg, text_embedder, status_callback=None):
    """
    Yield (embedding, user_rating) pairs for universal evaluator training.

    Args:
        cfg:             OmegaConf config object
        text_embedder:   the shared embedder — call text_embedder.embed_text(text),
                         which returns [n_chunks, dim]
        status_callback: Optional callable for progress reporting — status_callback("message")

    Yields:
        tuple of (torch.Tensor, float) — (text_embedding, user_rating)
    """
    import modules.my_module.db_models as db_models

    entries = db_models.MyModuleLibrary.query.filter(
        db_models.MyModuleLibrary.user_rating.isnot(None)
    ).all()

    for i, entry in enumerate(entries):
        if status_callback:
            status_callback(f"Processing {i + 1}/{len(entries)}")

        # Build a text description of the file (metadata, .meta content, etc.)
        description = build_description(entry)  # your logic here
        embedding = text_embedder.embed_text(description)

        yield embedding, entry.user_rating
```

The universal evaluator converts every rated item into a text description, embeds it, and trains on `(embedding, user_rating)` pairs. See [ARCHITECTURE.md](ARCHITECTURE.md) for the full training pipeline.
