# Module Architecture

How the Anagnorisis module system works under the hood: auto-discovery, initialization lifecycle, shared utilities, background task management, and the universal evaluator training pipeline.

---

## Auto-discovery lifecycle

When `app.py` starts, the following happens in order:

### Phase 0 — Config merging (before anything else)

```python
# app.py auto-merges module-specific config defaults into the root config:
for _mod_cfg_path in sorted(glob.glob('modules/*/config.defaults.yaml')):
    _mod_cfg = OmegaConf.load(_mod_cfg_path)
    cfg = OmegaConf.merge(_mod_cfg, cfg)  # root config overrides module defaults
```

If your module ships a `config.defaults.yaml`, its values become the fallback defaults. Any matching keys in the root `config.yaml` take priority. This lets modules declare sensible defaults that users can override without editing module files.

### Phase 1 — Scanning (synchronous, at import time)

```
modules/
├── _module_template/  ← SKIPPED (starts with _)
├── images/            ← discovered
├── music/             ← discovered
├── text/              ← discovered
├── train/             ← discovered
├── videos/            ← discovered
└── WebSearch/         ← discovered
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

Once initialized, the module's socket event handlers and Flask routes are live. The module responds to frontend events until the server shuts down. Scheduled background tasks (rating, description generation) continue running on daemon threads.

---

## File responsibilities

### `serve.py` — The entry point

**Must export:** `init_socket_events(socketio, app=None, cfg=None, data_folder='./project_data')`

This is the only function the framework calls. Parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `socketio` | `flask_socketio.SocketIO` | Shared Socket.IO instance for event handlers |
| `app` | `flask.Flask` | Flask app for registering routes; also exposes `app.task_manager` |
| `cfg` | `omegaconf.DictConfig` | Merged config from `config.yaml`, module `config.defaults.yaml` files, and overrides |
| `data_folder` | `str` | Legacy — prefer reading paths from `cfg` directly |

Typical initialization order inside `init_socket_events`:

```
CommonSocketEvents  →  read config  →  init engine  →  load evaluator
→  FileManager  →  metadata search  →  CommonFilters  →  Flask routes
→  socket handlers  →  schedule background tasks
```

### `db_models.py` — Database models *(optional)*

Only needed if the module persists per-file data. The framework handles its absence gracefully — `app.py` simply skips the import if the file doesn't exist.

**Must import `db` from `src.db_models`** — never create a new `SQLAlchemy()` instance.

Tables are auto-created and auto-migrated via Flask-Migrate.

### Content search — shared, not per-module

Modules no longer ship an `engine.py`. One multimodal model embeds every kind of content, and `src.content_search.ContentSearch` serves all of them; `get_content_search(cfg, media_type)` hands you an instance scoped to your media type. It provides:

- **Model downloading** from HuggingFace Hub → local `models/` directory, on first use
- **Two-level caching** (in-memory + on-disk) in one shared `content` namespace, keyed by `(path, model_hash, version)`
- **File hashing** chosen per media type (MD5 by default; video is fingerprinted by sampling, so an 8 GB film is identified without reading 8 GB)
- **Batch processing** with progress callbacks
- **One instance per media type**, sharing a single process-wide embedder

### `page.html` — Frontend template

Raw HTML fragment injected into `base.html` at runtime:

```
base.html
├── <head> (Bulma CSS, jQuery, Socket.IO, MathJax)
├── <nav>  (navbar with module links, theme switcher, task manager badge)
└── {% block content %}
    └── YOUR page.html CONTENT HERE
    {% endblock %}
```

You do **not** write `<html>`, `<head>`, or `<body>` tags. The global `socket` variable is already available.

### `js/main.js` — Frontend logic

ES module loaded via `<script type="module">`. Has access to:
- `socket` — the Socket.IO client (global from `base.html`)
- Shared components importable from `/modules/`

### `train.py` — Universal evaluator contribution *(optional)*

Exposes `get_training_pairs(cfg, text_embedder, status_callback)` — a generator that yields `(chunk_embeddings, user_rating)` pairs from user-rated items. Auto-discovered by the training pipeline via `modules/*/train.py` glob.

### `config.defaults.yaml` — Module configuration defaults *(optional)*

Declares default configuration values for your module. Merged into the root config at startup with root `config.yaml` values taking priority. This lets modules ship sensible defaults without requiring users to manually edit `config.yaml`.

### `requirements.txt` — Module-specific Python dependencies *(optional)*

Lists additional pip packages your module needs beyond the core dependencies. Users must rebuild the Docker image (or `pip install -r`) after adding a module with extra dependencies.

### Additional files

Modules can include any extra `.py` files (e.g. `crawler.py` in the WebSearch module), additional `js/` files (e.g. `PlaylistManager.js`, `SongControlPanel.js` in the music module), and documentation (`README.md`, `CHANGELOG.md`).

---

## Shared Python utilities

| Module | Purpose |
|--------|---------|
| `src.socket_events.CommonSocketEvents` | Throttled status/progress broadcasting. Two methods: `show_loading_status()` (init phase) and `show_search_status()` (runtime). |
| `src.content_search.get_content_search` | Returns the shared content-search engine, scoped to a media type. Embeds and compares file *content*. Replaces the per-module engines. |
| `src.file_manager.FileManager` | Discovers files in `media_directory`, computes hashes, handles pagination, and coordinates with the search engine for embedding extraction. Also provides `get_unrated_files(evaluator_hash)` and `list_all_files()`. |
| `src.common_filters.CommonFilters` | Pluggable sorting/filtering system. Built-in filters: `by_text`, `by_file`, `file_size`, `similarity`, `random`, `rating`. Modules can add custom filters (e.g. `recommendation`, `length`). |
| `src.metadata.search.get_metadata_search` | Returns the process-wide `MetadataSearch`. Builds text descriptions from file metadata (name, path, EXIF/tags, OmniDescriptor captions, `.meta` sidecars) and embeds them for semantic search. Module-independent: it resolves a file's media type from its extension, so it needs nothing from your module. |
| `anagnorisis_core.media_types.get_registry` | The media-type taxonomy loaded from `anagnorisis_core/data/media_types/`. Answers "what kind of content is this file", which decides its extensions, tag vocabulary and internal-metadata reader. |
| `src.scoring_models.Evaluator` | Base neural network for scoring. The universal evaluator (`TransformerEvaluator`) is the preferred variant for cross-module rating. |
| `src.model_manager.ModelManager` | Wraps ML models for GPU memory-efficient inference with automatic device management and idle timeout. |
| `src.db_models.db` | The shared SQLAlchemy instance. All modules must import `db` from here. |
| `anagnorisis_core.embedder.get_omni_embedder` | The single embedding model — text, images, audio and video into one shared vector space. Runs in a subprocess to keep the CUDA context out of the Flask process, and unloads when idle. Used by background tasks. |
| `anagnorisis_core.embedder.get_query_embedder` | The same model on the CPU, in-process, for embedding search queries. Searching must never touch the GPU. |
| `anagnorisis_core.descriptor.OmniDescriptor` | Multi-modal captioning model that generates text descriptions from images, audio, video, or text files. |
| `src.task_manager.TaskManager` | Centralised background task queue accessible via `app.task_manager`. Tasks run sequentially with cooperative pause/resume/cancel via `TaskContext`. Progress is broadcast to the frontend `TaskManagerComponent`. |
| `src.scheduler.schedule_task` | Utility to run a function periodically on a daemon thread with `app.app_context()`. Used by all media modules for background rating and description generation. |
| `anagnorisis_core.caching.TwoLevelCache` | Tiered RAM/disk cache with deferred writes and sharded storage. Used internally by search engines and metadata search. |

## Shared JavaScript components

| Component | Import path | Purpose |
|-----------|-------------|---------|
| `SearchBarComponent` | `/modules/SearchBarComponent.js` | Text search input + sort/mode dropdown + temperature control |
| `FileGridComponent` | `/modules/FileGridComponent.js` | Responsive card grid for file previews |
| `PaginationComponent` | `/modules/PaginationComponent.js` | Page navigation controls |
| `FolderViewComponent` | `/modules/FolderViewComponent.js` | Collapsible folder tree sidebar |
| `StarRatingComponent` | `/modules/StarRating.js` | Clickable 1–10 star rating widget |
| `ContextMenuComponent` | `/modules/ContextMenuComponent.js` | Right-click context menu builder |
| `MetaEditor` | `/modules/MetaEditor.js` | File metadata editor modal — typically instantiated twice per module: one editable instance for `.meta` sidecar files and one read-only instance for viewing full AI-generated descriptions |
| `TaskManagerComponent` | `/modules/TaskManagerComponent.js` | Navbar badge + modal showing active/queued/history tasks with pause/resume/cancel controls (auto-initialised by `base.html`) |

---

## Task Manager

Anagnorisis includes a centralised background task queue (`src/task_manager.py`) accessible via `app.task_manager`. It provides:

- **Sequential execution** — tasks run one at a time in a single worker thread.
- **Cooperative pause/resume/cancel** — each task receives a `TaskContext` object with `ctx.check()` (raises if cancelled), `ctx.update(progress, message)` (throttled at 250 ms), and threading events for pause/resume.
- **Frontend visibility** — the `TaskManagerComponent` in `base.html` shows a navbar badge with active/queued task count and a modal with full task details. No module-side JS needed.

### Usage in `serve.py`

```python
# Submit a long-running task (e.g. batch rating)
def task(ctx):
    for i, file_path in enumerate(files):
        ctx.check()                                 # Raises if user cancelled
        ctx.update(i / len(files), f'Rating {i+1}/{len(files)}...')
        do_expensive_work(file_path)

app.task_manager.submit('My Module: rate files (42)', task)
```

### Checking for duplicate tasks

Before submitting, check whether a task with the same name is already active or queued:

```python
state = app.task_manager.get_state()
active = state['active']
if (active and active.get('name', '').startswith(base_name)) or \
        any(t.get('name', '').startswith(base_name) for t in state['queued']):
    return  # Already running or queued — skip
```

---

## Scheduled background tasks

Media modules use `src.scheduler.schedule_task()` to periodically check for work and submit it to the Task Manager. Two patterns are standard:

### Background rating

Periodically finds files that the universal evaluator hasn't rated yet and submits them as a Task Manager job:

```python
from src.scheduler import schedule_task

def _check_and_submit_rating():
    """Scheduled: find unrated files and submit a rating task."""
    # 1. Skip if a rating task is already active or queued
    # 2. candidates = file_manager.get_unrated_files(evaluator.hash)
    # 3. app.task_manager.submit('My Module: rate unrated files (N)', task)

rating_interval = OmegaConf.select(cfg, 'my_module.rating_update_interval_minutes', default=None)
schedule_task(app, interval_minutes=rating_interval, fn=_check_and_submit_rating)
```

### Background description generation

Your module does **not** schedule this. Descriptions and metadata embeddings are
filled by the app-wide `MetadataIndexer` (`src/metadata/indexer.py`), which makes
one pass over every configured server and covers every media type at once — so a
file is described and searchable whether or not a module owns its content kind.
Its intervals and batch sizes live in the `metadata_search:` section of
`config.yaml`.

Rating stays per-module, because only your module knows its evaluator.

Intervals are read from `config.yaml` (or `config.defaults.yaml`) and default to
`None` (disabled). The `schedule_task` utility is a no-op when the interval is falsy.

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
│  → file name + path + OmniDescriptor caption + internal metadata │
│    + {file_name}.meta custom descriptions                        │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                    Text embedding                                │
│  embedder.embed_text(description)                                │
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

The training pipeline (`modules/train/universal_train.py`) **auto-discovers** all modules with a `train.py` file via a `modules/*/train.py` glob (skipping `_`-prefixed folders). For each discovered module it calls:

```python
get_training_pairs(cfg, text_embedder, status_callback)
```

This generator yields `(chunk_embeddings, user_rating)` pairs that are collected and used to train the single shared evaluator.

To add your module to the universal evaluator:

1. **Ensure your `db_models.py`** has `user_rating` and `file_path` columns.
2. **Create `train.py`** in your module folder exposing `get_training_pairs(cfg, text_embedder, status_callback)`.
3. **Give your media type a `metadata_extractor`** in `media_types/media_types.yaml` if its files carry internal metadata worth reading — the reader itself goes in `anagnorisis_core/extractors/`. This is what `generate_full_description()` uses to build the text representation.

That's it. No registry entries needed — the pipeline finds your `train.py` automatically.

### The `get_training_pairs` interface

```python
def get_training_pairs(cfg, text_embedder, status_callback=None):
    """
    Yields (chunk_embeddings: np.ndarray[chunks, dim], user_rating: float)
    
    - cfg: merged OmegaConf config
    - text_embedder: the shared embedder; embed_text(text) → np.ndarray[chunks, dim]
    - status_callback: optional callable(str) for progress reporting
    """
```

Two embedding strategies are commonly used:
- **"metadata"** (images, music, videos) — generate a text description via `MetadataSearch.generate_full_description()`, then embed that text.
- **"full_text"** (text, WebSearch) — read the file content directly and embed it.

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

This text is then embedded by the shared embedder and used as the training input. The key insight is that **all media types are unified into text** before training, which is why a single evaluator works across images, audio, video, and documents.

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
