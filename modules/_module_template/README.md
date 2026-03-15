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
#    (in serve.py, db_models.py, engine.py, page.html, js/main.js)

# 3. Add a config section in config.yaml
# 4. Restart the application — done!
```

---

## Module structure

```
modules/my_module/
├── serve.py          # REQUIRED — entry point, socket events, Flask routes
├── engine.py         # REQUIRED for media modules — search / embedding engine
├── page.html         # REQUIRED — frontend HTML (injected into base.html)
├── js/
│   └── main.js       # REQUIRED — frontend JavaScript (ES module)
├── db_models.py      # OPTIONAL — only if the module needs persistent storage (ratings, play counts, etc.)
├── train.py          # OPTIONAL — hook for the universal evaluator training pipeline
└── README.md         # OPTIONAL — module-specific documentation
```

---

## Step-by-step guide

### 1. Implement the search engine (`engine.py`)

Subclass `BaseSearchEngine` and implement the abstract methods:

- `model_name` — HuggingFace model ID (or `None`)
- `cache_prefix` — unique string for your cache files
- `_load_model_and_processor(local_model_path)` — load the ML model
- `_process_single_file(file_path)` — produce a `torch.Tensor` embedding
- `_get_metadata(file_path)` — return a dict of file metadata
- `_get_db_model_class()` — return your SQLAlchemy model class

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
import SearchBarComponent  from '/modules/SearchBarComponent.js';
import FileGridComponent   from '/modules/FileGridComponent.js';
import PaginationComponent from '/modules/PaginationComponent.js';
import StarRatingComponent from '/modules/StarRating.js';
```

### 5. Add configuration (`config.yaml`)

```yaml
my_module:
  embedding_model: "namespace/model-name"
  media_directory: /path/to/media/files
  media_formats:
    - .ext1
    - .ext2
```

### 6. Integrate with the universal evaluator (`train.py`) — *optional*

Anagnorisis does **not** use per-module evaluator models. Instead, a single **universal evaluator** is trained on user-rated items from **all** modules. It works by converting every media file into a text description (via `MetadataSearch.generate_full_description()`), embedding that text, and training on `(embedding, user_rating)` pairs.

To integrate your module, create `train.py` that exposes a helper the training system can call:

```python
import os

def get_rated_items(cfg):
    """
    Return a list of dicts for the universal evaluator training pipeline.

    Each dict must contain:
      - "text_path": str — absolute path to a text file (.txt / .md) containing
                           a textual representation of the media item
                           (auto-generated description, metadata, etc.)
      - "user_rating": float — the user's score for this item (0–10)

    The universal evaluator embeds the text and trains on (embedding, rating) pairs.
    """
    import modules.my_module.db_models as db_models

    entries = db_models.MyModuleLibrary.query.filter(
        db_models.MyModuleLibrary.user_rating.isnot(None)
    ).all()

    media_dir = cfg.my_module.media_directory
    rated_items = []

    for entry in entries:
        file_path = os.path.join(media_dir, entry.file_path)
        # Text description file — generated by MetadataSearch or your own logic
        text_path = file_path + ".description.txt"

        if os.path.isfile(text_path):
            rated_items.append({
                "text_path": text_path,
                "user_rating": entry.user_rating,
            })

    return rated_items
```

The text description file should mirror what `MetadataSearch.generate_full_description()` produces: file name, relative path, OmniDescriptor auto-description, internal metadata, and `.meta` sidecar content. See [ARCHITECTURE.md](ARCHITECTURE.md) for the full training pipeline.
