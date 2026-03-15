# Module Architecture

How the Anagnorisis module system works under the hood: auto-discovery, initialization lifecycle, shared utilities, and the universal evaluator training pipeline.

---

## Auto-discovery lifecycle

When `app.py` starts, the following happens in order:

### Phase 1 — Scanning (synchronous, at import time)

```
modules/
├── _module_template/  ← SKIPPED (starts with _)
├── images/            ← discovered
├── music/             ← discovered
├── text/              ← discovered
├── train/             ← discovered
└── videos/            ← discovered
```

1. `app.py` lists all directories in `modules/` whose name does **not** start with `_`.
2. For each directory, if `db_models.py` exists, it is imported and its SQLAlchemy model classes are registered with Flask-Migrate. This ensures database tables are created/migrated before any module code runs.
3. A URL route `/<module_name>` is registered immediately — but it initially returns the **loading screen** (`modules/loading.html`).

### Phase 2 — Initialization (background thread, sequential)

A single background thread iterates through all discovered modules that have a `serve.py`:

```python
for extension_name in valid_extensions:
    background_init_extension(app, socketio, cfg, data_folder, extension_name)
```

For each module:
1. `serve.py` is imported.
2. `init_socket_events(socketio, app, cfg, data_folder)` is called.
3. This is where your module does all heavy work: loading ML models, building caches, registering socket handlers.
4. `show_loading_status()` calls during this phase are forwarded to the loading screen in real-time.
5. When `init_socket_events` returns, the module is marked as ready and the URL route starts serving `page.html` instead of the loading screen.

> **Important:** Modules are initialized **sequentially** in a single thread to avoid import race conditions. A slow module delays all modules after it.

### Phase 3 — Runtime

Once initialized, the module's socket event handlers and Flask routes are live. The module responds to frontend events until the server shuts down.

---

## File responsibilities

### `serve.py` — The entry point

**Must export:** `init_socket_events(socketio, app=None, cfg=None, data_folder='./project_data')`

This is the only function the framework calls. Parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `socketio` | `flask_socketio.SocketIO` | Shared Socket.IO instance for event handlers |
| `app` | `flask.Flask` | Flask app for registering routes |
| `cfg` | `omegaconf.DictConfig` | Merged config from `config.yaml` and overrides |
| `data_folder` | `str` | Legacy — prefer reading paths from `cfg` directly |

Typical initialization order inside `init_socket_events`:

```
CommonSocketEvents  →  read config  →  init engine  →  FileManager
→  MetadataSearch  →  CommonFilters  →  Flask routes  →  socket handlers
```

### `db_models.py` — Database models *(optional)*

Only needed if the module persists per-file data. The framework handles its absence gracefully — `app.py` simply skips the import if the file doesn't exist.

**Must import `db` from `src.db_models`** — never create a new `SQLAlchemy()` instance.

Tables are auto-created and auto-migrated via Flask-Migrate.

### `engine.py` — Search / embedding engine

Subclass of `src.base_search_engine.BaseSearchEngine`. The base class provides:

- **Model downloading** from HuggingFace Hub → local `models/` directory
- **Two-level caching** (in-memory LRU + on-disk pickle) keyed by `(file_hash, model_hash)`
- **File hashing** with configurable algorithm
- **Batch processing** with progress callbacks
- **Singleton pattern** — one instance per subclass, shared across the app

### `page.html` — Frontend template

Raw HTML fragment injected into `base.html` at runtime:

```
base.html
├── <head> (Bulma CSS, jQuery, Socket.IO, MathJax)
├── <nav>  (navbar with module links)
└── {% block content %}
    └── YOUR page.html CONTENT HERE
    {% endblock %}
```

You do **not** write `<html>`, `<head>`, or `<body>` tags. The global `socket` variable is already available.

### `js/main.js` — Frontend logic

ES module loaded via `<script type="module">`. Has access to:
- `socket` — the Socket.IO client (global from `base.html`)
- Shared components importable from `/modules/`

---

## Shared Python utilities

| Module | Purpose |
|--------|---------|
| `src.socket_events.CommonSocketEvents` | Throttled status/progress broadcasting. Two methods: `show_loading_status()` (init phase) and `show_search_status()` (runtime). |
| `src.base_search_engine.BaseSearchEngine` | Abstract base for all embedding engines. Provides model management, caching, and batch processing. |
| `src.file_manager.FileManager` | Discovers files in `media_directory`, computes hashes, handles pagination, and coordinates with the search engine for embedding extraction. |
| `src.common_filters.CommonFilters` | Pluggable sorting/filtering system. Built-in filters: `by_text`, `by_file`, `file_size`, `similarity`, `random`, `rating`. Modules can add custom filters. |
| `src.metadata_search.MetadataSearch` | Builds text descriptions from file metadata (name, path, EXIF/tags, OmniDescriptor captions, `.meta` sidecars) and embeds them for semantic search. |
| `src.scoring_models.Evaluator` | Base neural network for scoring. The universal evaluator (`TransformerEvaluator`) is the preferred variant. |
| `src.model_manager.ModelManager` | Wraps ML models for GPU memory-efficient inference with automatic device management. |
| `src.db_models.db` | The shared SQLAlchemy instance. All modules must import `db` from here. |
| `src.text_embedder.TextEmbedder` | Converts text strings to embedding vectors using the configured text model (e.g. jina-embeddings-v3). |
| `src.omni_descriptor.OmniDescriptor` | Multi-modal captioning model that generates text descriptions from images, audio, video, or text files. |

## Shared JavaScript components

| Component | Import path | Purpose |
|-----------|-------------|---------|
| `SearchBarComponent` | `/modules/SearchBarComponent.js` | Text search input + sort dropdown |
| `FileGridComponent` | `/modules/FileGridComponent.js` | Responsive card grid for file previews |
| `PaginationComponent` | `/modules/PaginationComponent.js` | Page navigation controls |
| `FolderViewComponent` | `/modules/FolderViewComponent.js` | Collapsible folder tree sidebar |
| `StarRatingComponent` | `/modules/StarRating.js` | Clickable 1–10 star rating widget |
| `ContextMenuComponent` | `/modules/ContextMenuComponent.js` | Right-click context menu |
| `MetaEditor` | `/modules/MetaEditor.js` | File metadata editor modal |

---

## Universal evaluator training pipeline

Anagnorisis uses a **single universal evaluator** instead of per-module scoring models. Here is how it works:

### Overview

```
┌───────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│  images   │     │  music   │     │  videos  │     │   text   │
│ (rated    │     │ (rated   │     │ (rated   │     │ (rated   │
│  files)   │     │  files)  │     │  files)  │     │  files)  │
└────┬──────┘     └────┬─────┘     └────┬─────┘     └────┬─────┘
     │                 │                │                │
     ▼                 ▼                ▼                ▼
┌──────────────────────────────────────────────────────────────────┐
│                  Text description generation                     │
│  MetadataSearch.generate_full_description()                      │
│  → file name + path + OmniDescriptor caption + internal metadata |
|    + {file_name}.meta custom descriptions                        │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                    Text embedding                                │
│  TextEmbedder.embed_text(description)                            │
│  → chunk into tokens → embed each chunk → list of vectors        │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│               TransformerEvaluator training                      │
│  train on (embedding_chunks, user_rating) pairs                  │
│  → saves to project_config/models/universal_evaluator.pt         │
└──────────────────────────────────────────────────────────────────┘
```

### How modules participate

The training pipeline (`modules/train/universal_train.py`) has a `_MODULE_DEFS` registry that lists all modules to gather rated files from. Each entry specifies:

- `db_import` / `db_class` — where to find rated entries
- `engine_import` / `engine_class` — the search engine (needed for file hashing and metadata)
- `embedding_method` — `"metadata"` (generate text description) or `"full_text"` (use raw file content for text files)

To add your module to the universal evaluator:

1. **Ensure your `db_models.py`** has `user_rating` and `file_path` columns.
2. **Add an entry to `_MODULE_DEFS`** in `modules/train/universal_train.py`:

```python
{
    "name": "my_module",
    "config_attr": "my_module",
    "db_import": "modules.my_module.db_models",
    "db_class": "MyModuleLibrary",
    "engine_import": "modules.my_module.engine",
    "engine_class": "MyModuleSearch",
    "embedding_method": "metadata",  # or "full_text" for text-like content
},
```

3. **Ensure `engine.py` implements `_get_metadata()`** — this is what `MetadataSearch.generate_full_description()` calls to build the text representation.

### The text description format

`MetadataSearch.generate_full_description()` builds a text like:

```
File Name: sunset_beach.jpg
File Path: vacation/2024/sunset_beach.jpg

# Automatic description:
A beautiful sunset over a tropical beach with palm trees silhouetted
against an orange and purple sky. Calm ocean waves in the foreground.

# Internal metadata:
format: JPEG
mode: RGB
width: 4032
height: 3024
resolution: 4032x3024
Make: Apple
Model: iPhone 15 Pro

# External metadata from 'sunset_beach.jpg.meta' file:
Location: Maui, Hawaii
Tags: vacation, sunset, beach, landscape
```

This text is then embedded by `TextEmbedder` and used as the training input. The key insight is that **all media types are unified into text** before training, which is why a single evaluator works across images, audio, video, and documents.

### Optional: `train.py` helper

Instead of modifying `universal_train.py` directly, you can create a `train.py` in your module that exposes a `get_rated_items(cfg)` function. This returns a list of dicts with `text_path` (path to a pre-generated text description file) and `user_rating` (the user's score). See [README.md](README.md) for the code template.

---

## Data flow diagram

```
User action in browser
        │
        ▼
  main.js: socket.emit('emit_{module}_page_{action}', data)
        │
        ▼
  serve.py: @socketio.on('emit_{module}_page_{action}')
        │
        ├──→ FileManager.get_files(...)
        │         │
        │         ├──→ engine.process_files(...)  → embeddings
        │         ├──→ engine.get_file_hash(...)  → content hash
        │         └──→ CommonFilters.filter_by_*  → sorted file list
        │
        ├──→ db_models.MyModuleLibrary.query...  → DB read/write
        │
        └──→ return response  →  socket response to client
                                        │
                                        ▼
                                  main.js: socket.on(...)
                                        │
                                        ▼
                              FileGridComponent.render(data)
```
