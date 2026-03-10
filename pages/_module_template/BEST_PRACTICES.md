# Module Development Best Practices

Code style, naming conventions, common pitfalls, and tips for building robust Anagnorisis modules.

---

## Naming conventions

Consistent naming is what makes auto-discovery and cross-module compatibility work.

| What | Convention | Example |
|------|-----------|---------|
| Folder name | `snake_case`, **no** `_` prefix | `pages/my_module/` |
| Config section | Same as folder name | `my_module:` in `config.yaml` |
| Socket events | `emit_{module}_page_{action}` | `emit_my_module_page_get_files` |
| DB model class | `PascalCase` + `Library` | `MyModuleLibrary` |
| Search engine class | `PascalCase` + `Search` | `MyModuleSearch` |
| `CommonSocketEvents` module_name | Same as folder name | `CommonSocketEvents(socketio, module_name="my_module")` |
| Flask route for raw files | `/{module}_files/<path:filename>` | `/my_module_files/photo.jpg` |
| CSS status class | `.{module}-search-status` | `.my_module-search-status` |
| Cache prefix | Same as folder name | `cache_prefix = 'my_module'` |

> **Why it matters:** The `FileManager`, `CommonFilters`, loading screen, and status broadcasting all derive names from `module_name`. A mismatch causes silent failures.

---

## Python code style

### Keep `serve.py` focused

`serve.py` should be an orchestrator, not a monolith. It should:
- Read config and set up dependencies
- Define thin socket event handlers that delegate to engine / file manager
- Register Flask routes

Heavy logic (embedding, metadata extraction, file processing) belongs in `engine.py` or separate utility files within your module folder.

### Use `CommonSocketEvents` for all user-facing status

```python
# Good — throttled, broadcasts to clients correctly
common_socket_events.show_search_status(f"Processing {i}/{total} files...")

# Bad — unthrottled, floods the client
socketio.emit('emit_show_search_status', f"Processing {i}/{total} files...")
```

`show_loading_status()` is for the initialisation phase (shown on the loading screen).  
`show_search_status()` is for runtime operations (shown in the status bar).

### Use `nonlocal` for mutable closures in route handlers

```python
def init_socket_events(socketio, app=None, cfg=None, data_folder='./project_data'):
    media_directory = cfg.my_module.media_directory

    @app.route('/my_module_files/<path:filename>')
    def serve_files(filename):
        nonlocal media_directory  # Required if media_directory can change
        return send_from_directory(media_directory, filename)
```

### Handle missing config gracefully

```python
# Good — module still loads, just without media browsing
media_directory = cfg.get("my_module", {}).get("media_directory", None)
if media_directory is None:
    print("My module: media folder is not set.")

# Bad — crashes the entire application
media_directory = cfg.my_module.media_directory  # KeyError if section missing
```

### Avoid top-level side effects

Imports in `serve.py` and `engine.py` run when the module is discovered. Don't load models, open files, or start threads at import time — do it inside `init_socket_events()` or `initiate()`.

### Use the app context for database operations in background threads

```python
# When running DB queries outside a request context (e.g. during init):
with app.app_context():
    entries = MyModuleLibrary.query.filter_by(user_rating=None).all()
```

---

## Database model guidelines

### When to create `db_models.py`

Create it **only** if your module needs to persist per-file data across restarts:
- User ratings
- Play counts / skip counts
- Model ratings from the universal evaluator
- Module-specific metadata that is expensive to recompute

If your module is purely functional (e.g. a settings page, a visualisation tool), **skip `db_models.py` entirely**.

### Standard columns

If you do create a DB model, keep these columns for framework compatibility:

| Column | Type | Purpose |
|--------|------|---------|
| `id` | Integer, primary key | Row identity |
| `hash` | String, unique | Content-based file identifier |
| `hash_algorithm` | String | Records which hashing method was used |
| `file_path` | String | Path relative to `media_directory` |
| `user_rating` | Float | User's score (0–10) |
| `user_rating_date` | DateTime | When the user last rated this file |
| `model_rating` | Float | Universal evaluator's predicted score |
| `model_hash` | String | Hash of the evaluator checkpoint that produced `model_rating` |

These columns are expected by `FileManager`, `CommonFilters`, and the CSV export/import system.

### Always implement `as_dict()`

```python
def as_dict(self):
    return {c.name: getattr(self, c.name) for c in self.__table__.columns}
```

This is used by the CSV export and various internal serialisation paths.

---

## Frontend best practices

### IIFE wrapping

Wrap all module JS in an IIFE to avoid global namespace pollution:

```javascript
(function () {
  const MODULE_NAME = 'my_module';
  // ... all module code here ...
})();
```

### Preserve URL state

Use URL parameters for page number, path, sort, and seed so that browser navigation (back/forward) works correctly:

```javascript
const urlParams = new URLSearchParams(window.location.search);
let page = parseInt(urlParams.get('page')) || 1;
let path = decodeURIComponent(urlParams.get('path') || '');
```

### Use shared components instead of re-implementing

The `pages/` directory contains battle-tested UI components. Don't rebuild search bars, pagination, file grids, or star ratings — import and configure the existing ones.

### Load JS as ES modules

```html
<!-- Good -->
<script type="module" src="pages/my_module/js/main.js"></script>

<!-- Bad — pollutes global scope, no import support -->
<script src="pages/my_module/js/main.js"></script>
```

### Keep CSS in `page.html`

Use `<style>` blocks inside `page.html` for module-specific styles. Don't create separate CSS files unless the styles are substantial — it reduces the number of files and keeps everything co-located.

---

## Search engine best practices

### Always use the singleton pattern

`BaseSearchEngine` implements `__new__` to enforce a singleton per subclass. Don't fight it — the same engine instance is shared between `serve.py`, `CommonFilters`, and the training pipeline.

### Keep `_process_single_file` deterministic

Given the same file, always return the same embedding. The caching layer relies on `(file_hash, model_hash)` as cache keys.

### Let `_get_metadata` be fast

Metadata extraction is called frequently (every file on every page load). Avoid heavy I/O or computation. If extraction is expensive, cache the results.

### Use `_get_model_hash_postfix` for versioning

When you change how embeddings are computed (different preprocessing, different pooling), bump the postfix so cached embeddings are invalidated:

```python
def _get_model_hash_postfix(self):
    return "_v1.1.0"  # Bump this when embedding logic changes
```

---

## Common pitfalls

| Pitfall | Solution |
|---------|----------|
| Module folder starts with `_` | Rename — `_` prefix folders are excluded from auto-discovery |
| Socket events don't reach the frontend | Check that event names match exactly between `serve.py` and `main.js` |
| Database tables not created | Ensure `db_models.py` imports `db` from `src.db_models`, not a new `SQLAlchemy()` instance |
| Module crashes on startup | Check the Docker/app logs; `init_socket_events` runs in a background thread, errors may be swallowed |
| Files 404 in the browser | Verify the Flask route path matches what the frontend requests (e.g. `/my_module_files/...`) |
| Stale embeddings after model change | Bump `_get_model_hash_postfix()` to invalidate caches |
| `model_name` returns `None` but you expected it | Check that your config section name in `config.yaml` exactly matches what `engine.py` reads |

---

## Testing your module

1. **Start simple.** Get file listing working before adding embeddings.
2. **Set `model_name = None`** to skip model downloading during initial development.
3. **Check the loading screen.** Visit `/<module_name>` while the app is starting — you should see progress messages from `show_loading_status()`.
4. **Watch the logs.** Any exceptions in `init_socket_events` are caught and logged, but the module will appear stuck on the loading screen.
5. **Test with a small media folder** first — a handful of files is enough to validate the full pipeline.
6. **Test the rating flow end-to-end:** rate a file → check it appears in the database → train the universal evaluator → verify `model_rating` gets written back.
