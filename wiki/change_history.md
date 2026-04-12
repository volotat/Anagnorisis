# Change History


## TODO list
 
### Music 
Add a way to change the rest of the metadata of the song.  
Add volume control.   
Move the "edit" button to the song list element.  
Add "Chain Mode" where each song is selected as the most similar to a previous song.  

### Images 
Implement some sort of effective resolution estimation technique.   
Improve sorting by resolution performance (one idea might be caching information of image resolution).   
Add a way to select only some particular folders for displaying images from.  
Remove from DB images that no longer exist in the media folder and have no user rating. This should reduce the size of the DB as embeddings took quite a lot of space.  
Make number of images per page adjustable via config.yaml file.  
Make number of columns of images presented dependent on the screen size. 
Add BRISQUE score filter for simple image quality assessment. 
Replace SigLIP model with more powerful MobileClip2.

### Train page
Disable the start button if fine-tuning has already started.  
When refreshing the page the information about the previous run should appear.  
Pressing the start button should remove all the information from the graph and restart the process.  
Add a button to stop the training process and save the best model so far.  
After training is complete or canceled reload the evaluation models.  

### Wiki
Complete self-hosting guide.

### General
Add some sort of a control panel that is active on all pages (may be hidden) and shows current GPU memory load by the system and console output.  
Create an extension that provides recent Arxiv papers (https://arxiv.org/list/cs.LG/recent).    
Implement usage of FAISS library for fast vector search.    
When the folder name contains '[' and ']' symbols it is not correctly read by python scripts for some reason.   
Add automatic database backup generation from time to time to prevent loss of data in case of a failure.  
Find a way for more optimal embeddings storage in the DB.  
Add display of the current version of the project into the header. 
Add pop-up messages about new version of the project available to download and some button that allows to download it directly from the UI and restart the server.
When starting the Docker container, if the provided path to the share folder does not exists the error should be displayed and the container should not start.
Add a way to create new folders in the UI.  
Add a way to copy path to a folder trough the UI.  
When gathering hashes of files make the cache saving once in a while, as this process might take quite a long time especially for video files.
All active filters should be automatically gathered from the backend and displayed in the UI, instead of being hardcoded in the frontend.
In the recommendation engine replace `full_play_count` and `skip_count` with `play_time` that dynamically calculate how much time user spent listening or watching the file. This should better reflect user preferences and be consistent across different types of data (at least music and videos).

Replace all the hash gathering occurrences with a single function that handles it in a unified way and notify the UI about the progress.
Try to build a safe fail when the hashing algorithm is changes so we don't lose data in the DB that is hash related.

### Ideas
Add new downloadable module for 'Deep Research'-like functionality that uses user-trained text evaluating model to search for relative information adjusted to user preferences.

## Versions History

### Version 0.3.13 (12.04.2026)
*  **File Manager:**
    *   Fixed bug displaying error when trying to show files in the empty folder. Now the message "No files found in this folder" is displayed instead of an error.
*  **Docker Deployment:**
    *   If error occurs during the installation of the dependencies of the container (for example, due to incorrect media folder path), the error message is now displayed in the console.

### Version 0.3.12 (10.04.2026)
*   **Module Template:**
    *   Updated `modules/_module_template/` to match the actual module architecture. Updated `serve.py`, `engine.py`, `train.py`, `db_models.py`, `page.html`, `js/main.js`, `config.defaults.yaml`, `ARCHITECTURE.md`, `BEST_PRACTICES.md`, and `README.md` with accurate examples, conventions, and documentation reflecting the current codebase.
*   **Code Deduplication:**
    *   Extracted shared Python helpers (`src/module_helpers.py`) for `.meta` file handlers and scheduled background tasks (rating and description checks). All four modules now use these shared helpers instead of maintaining their own near-identical copies.
    *   Extracted shared JavaScript factory (`modules/ModuleMetaEditors.js`) for creating MetaEditor instances. Images, Music, and Videos modules now use a single `createModuleMetaEditors(socket, moduleName)` call instead of duplicating the same MetaEditor setup code.
*   **Dead Code Removal:**
    *   Removed unused `parse_terminal_command()` function from `src/file_manager.py`.
    *   Removed unused imports across all module `serve.py` files.
    *   Removed commented-out code blocks from Python and JavaScript files across all modules.

### Version 0.3.11 (08.04.2026)
*   **Background Ratings:**
    *   Replaced the DB-first unrated file query (which was counting stale rows pointing to deleted/moved files as "unrated") with a disk-first approach. `FileManager.get_unrated_files(evaluator_hash)` now walks the media directory, hashes every file via the engine's path+mtime cache, and queries the DB to find which hashes lack a current rating — naturally ignoring files that no longer exist on disk.
    *   `get_unrated_files()` calls `sync_file_paths()` first, so moved or renamed files have their DB `file_path` corrected before the rating lookup. Both operations share the same disk walk and hash pass, so there is no extra I/O.
    *   All four module `_check_and_submit_rating()` functions simplified: `candidates = file_manager.get_unrated_files(evaluator_hash)` replaces the `unrated_q()` factory, `sync_file_paths()` call, and `os.path.exists()` filter that were needed before.
*   **Background Description Generation:**
    *   Added proactive background description generation via `OmniDescriptor`, analogous to the existing rating scheduler. Every 10 minutes each module checks for files whose auto-description is not yet in the cache and submits a batch to the task manager.
    *   `MetadataSearch.get_undescribed_files(file_paths)` probes the shared `TwoLevelCache` for each file's `auto_desc::` key using the current model hash. Returns `None` if the model hash is not yet known (model was never loaded since install), in which case the scheduler falls back to treating all files as candidates.
    *   `MetadataSearch._get_omni_model_hash()` resolves the model hash from the proxy's in-memory `model_hash` first, then falls back to `omni_model_hash::{model_name}` in the shared cache. The hash is persisted to cache immediately after `OmniDescriptor.initiate()` completes, so it survives process restarts without reloading the model.
    *   `FileManager.list_all_files()` added — returns all media file paths on disk via the existing mtime-cached walker, without hashing. Used by the description scheduler (no DB involved).
    *   All four modules (`images`, `music`, `text`, `videos`) have a new `_check_and_submit_description()` function and a corresponding `schedule_task` registration at module startup.
    *   `config.yaml` gains `description_update_interval_minutes` (default: 10) and `description_update_batch_size` (default: 100; 50 for videos) per module. Set to `null` to disable.

### Version 0.3.10 (05.04.2026)
*   **OmniDescriptor:**
    *   Now new, optimized version of the `MiniCPM-o-4.5` model is used for generating descriptions. New model is provided by [Dystrio](https://huggingface.co/dystrio/MiniCPM-o-4_5-Sculpt-Throughput) and is pruned specifically for fast description generation and lower memory footprint, providing faster token generation speed and bigger context window.
*   **Task Manager:**
    *   Added a centralised background task queue (`src/task_manager.py`) with a single worker thread. Tasks run sequentially; each task receives a `TaskContext` that supports cooperative pause, resume, and cancel via `threading.Event`, and throttled progress updates (`ctx.update(progress, message)`) at 250 ms intervals.
    *   Added `TaskManagerComponent.js` — an IIFE frontend component that renders a navbar badge showing the number of active/queued tasks, and a Bulma modal with running, queued, and history sections. Individual tasks can be paused, resumed, cancelled, or removed from history directly in the UI.
    *   `TaskManager` is instantiated in `app.py` immediately after `SocketIO` and exposed as `app.task_manager` so all modules can reach it.
*   **Training — Background Processing:**
    *   Migrated music evaluator, image evaluator, and universal evaluator training handlers to `task_manager.submit()`. The `TRAINING_ACTIVE` global flag is replaced by inspecting the task queue for active/queued `Train:` tasks. Training progress is reported via `TaskContext` in addition to the existing Train-page chart socket events.
    *   Fixed a pre-existing bug in the training callback where `socketio.emit("emit_train_page_display_train_data", data)` was called unconditionally even when the throttle condition was false, potentially referencing an undefined `data` variable.
*   **Proactive Background Rating:**
    *   Each media module now periodically scans the DB for files with a missing or stale model rating and submits them to the Task Manager in the background. By the time a user browses to a folder its files are typically already rated, making the inline rating step faster.
    *   The check interval is configured per-module in `config.yaml` via `rating_update_interval_minutes` (set to `null` to disable).
    *   Added `src/scheduler.py` with a `schedule_task(app, interval_minutes, name, fn)` utility used by all four modules.
    *   Added `FileManager.sync_file_paths()` which walks the media directory on disk, compares file hashes against the DB, and corrects any stale `file_path` values (e.g. from moved files). Returns the number of rows updated. Each module's `_task_preemptive_rating` calls this first to ensure paths are current, then runs its own DB query to find files with a missing or outdated model rating.

### Version 0.3.9 (23.03.2026)
*   **Color Themes:**
    *   Added three color themes: **Light** (default), **Dark**, and **Solarized Light**. The active theme is persisted in `localStorage` and applied before first paint to avoid flash of unstyled content.
    *   Theme switcher dropdown added to the navbar, accessible on all pages.
    *   All hardcoded colors across HTML templates and JavaScript components replaced with Bulma CSS variables so they respond correctly to theme changes.

### Version 0.3.8 (21.03.2026)
*   **Universal Evaluator Training**
    *   Added modal window that controls how much time you want to spend on training the model. You can either set a number minutes to train the model or number of optimization steps to take.
    *   Migrated all built-in modules (music, images, videos, text) to the same self-contained training data interface. Each module now owns a `train.py` with a `get_training_pairs(cfg, text_embedder, status_callback)` generator that handles its own DB queries, file reading, and embedding — no central registry needed.
    *   Added `get_training_pairs()` to `modules/music/train.py` and `modules/images/train.py` (metadata strategy: generate a text description via `MetadataSearch` and embed with `text_embedder`).
    *   Created `modules/videos/train.py` with the same metadata strategy.
    *   Created `modules/text/train.py` supporting two strategies switchable via `cfg.evaluator.text_embedding_method`: `"full_text"` (default) embeds the raw file content directly; `"metadata"` generates a short description via `MetadataSearch` instead, useful if you want the evaluator to reason about summaries rather than raw content. Switching strategies requires no code changes, just a config update.
    *   Removed the legacy `_MODULE_DEFS` registry, `_LEGACY_MODULE_NAMES` exclusion list, `_import_attr` helper, `_gather_rated_files()`, and the two-phase embedding pipeline (Phase A full-text pre-computation + Phase B per-file metadata loop) from `universal_train.py`. All of that complexity now lives inside each module's own `train.py`.
    *   `_gather_from_module_train_files()` now covers all modules without any exclusions. Adding a new module to universal evaluator training is as simple as dropping a `train.py` with `get_training_pairs()` into its folder — no changes to core training code ever required.
    *   Updated `modules/_module_template/train.py` documentation to reflect the removal of the legacy exclusion list.

### Version 0.3.7 (15.03.2026)
*   **Universal Evaluator - Training Improvements:**
    *   Added three toggleable training augmentation strategies at the top of `universal_train.py`: **nonsensical negatives** (`ENABLE_NONSENSICAL_NEGATIVES`) injects random-noise embeddings mapped to score 0 to teach the model that meaningless content should rank lowest; **oversampling** (`ENABLE_OVERSAMPLING`) duplicates underrepresented score bins up to the median bin count; **loss weighting** (`ENABLE_LOSS_WEIGHTING`) applies per-sample inverse-frequency weights so rare scores contribute proportionally more to the gradient.
    *   Fixed `TransformerEvaluator.reinitialize()` to fully rebuild the model from scratch via a new `_build_transformer_net()` helper. The previous implementation iterated `nn.Linear` / `nn.LayerNorm` submodules and called `reset_parameters()`, but `nn.MultiheadAttention` stores its Q/K/V projections as raw `nn.Parameter` tensors that are not `nn.Linear` instances, so they survived every reinitialize call and carried weights from any previously loaded checkpoint into the new training run.
    *   Fixed `ModelManager` idle-eviction firing mid-epoch. The cleanup thread checks a `_busy` flag, but that flag was only held during each individual forward pass. Between batches (`loss.backward()` → next forward) `_busy` was `False`, so the thread could move the model to CPU mid-epoch, corrupting the computation graph and causing the periodic sharp accuracy drops visible in the training graph. The flag is now set for the entire epoch loop and cleared in a `finally` block.
*   **OmniDescriptor - Inference Benchmarks (`src/omni_descriptor.py`):**
    *   Added GPU inference speed benchmark (`benchmark_inference_speed()`) with a shared `_run_speed_probes()` core that runs two passes per probe - a prefill-only pass (`max_new_tokens=1`) and a full generation pass - to separately measure prefill throughput and generation throughput.
    *   Added CPU inference speed benchmark (`benchmark_inference_speed_cpu()`). The model is temporarily loaded in `bfloat16` (≈16 GB RAM) via a new `_with_cpu_model(fn)` helper that swaps the GPU model out, loads a CPU copy with `init_vision=True` and `init_audio=True`, calls the benchmark, then restores the GPU model. `torch.set_num_threads(os.cpu_count())` is set to utilise all available CPU cores.
    *   Added CPU context window benchmark (`benchmark_context_window_cpu()`) that probes increasing input sizes (128 → 4096 tokens) with RAM tracking via `psutil` and `resource.getrusage`, measuring peak RSS and free RAM at each step.
    *   Fixed overly conservative context window recommendations: the input-token suggestion was clamped to the last successful probe size even when no probe had failed; added a `hit_limit` flag so clamping only applies when a probe actually errors or OOMs. Also removed a hardcoded `min(..., 1024)` cap on suggested output tokens, letting the 80/20 input/output split determine the value naturally.
*   **Project Structure - Folder Rename (`pages/` to `modules/`):**
    *   Renamed the root `pages/` folder to `modules/` to better reflect its purpose as a container for self-contained feature modules rather than web pages.
    *   Updated all references across the codebase: Python package imports (`from pages.X` → `from modules.X`), filesystem path strings in `app.py`, the Flask `template_folder` and `send_from_directory` calls, the `@app.route('/modules/...')` static-file route, JavaScript ES module imports (`/pages/ComponentName.js` → `/modules/ComponentName.js`), HTML `<script src="...">` attributes, `_MODULE_DEFS` string values in `universal_train.py`, the dynamic `importlib` paths in `universal_train.py` and `app.py`, the `Dockerfile` `COPY` instruction, and the test shell script in `tests/commands.sh`.

### Version 0.3.6 (10.03.2026)
*   **Module Self-Containment:**
    *   Each module now owns its own `requirements.txt` (Python dependencies) and `config.defaults.yaml` (default configuration values), placed directly inside the module folder.
    *   The `Dockerfile` was updated to auto-discover and install all `pages/*/requirements.txt` files at build time via a `find … -exec pip install` pattern - no more manually listing module dependencies in the root `requirements.txt`.
    *   `app.py` now auto-discovers and deep-merges each module's `config.defaults.yaml` at startup using `OmegaConf.merge`, giving modules sensible defaults without requiring the user to add anything to the root `config.yaml`. User-provided values always take precedence.
    *   Root `requirements.txt` and `config.yaml` now contain only core application dependencies and settings; all module-specific additions live inside the module folder.
*   **Module Template & Documentation:**
    *   Added `pages/_module_template/` - a fully documented reference implementation for building new modules, including `serve.py`, `db_models.py`, `engine.py`, `page.html`, `js/main.js`, and `train.py`.
    *   Added `pages/_module_template/README.md` - step-by-step guide for creating a new module from scratch.
    *   Added `pages/_module_template/ARCHITECTURE.md` - explains the auto-discovery lifecycle, file responsibilities, shared Python utilities, shared JavaScript components, and the universal evaluator training pipeline.
    *   Added `pages/_module_template/BEST_PRACTICES.md` - naming conventions, Python and frontend code style guidelines, database model guidelines, search engine best practices, and a common pitfalls table.
*   **Universal Evaluator Training - Extensible Plugin System:**
    *   `universal_train.py` now auto-discovers `pages/*/train.py` files and calls `get_training_pairs(cfg, text_embedder, status_callback)` on any module that exposes it, collecting `(chunk_embeddings, user_rating)` pairs and including them in the training set alongside the legacy modules.
    *   The four legacy modules (music, images, videos, text) are excluded from auto-discovery by name to avoid double-counting; their existing `_MODULE_DEFS` path is unchanged.
    *   New modules can contribute training data simply by adding a `train.py` with `get_training_pairs()` - no changes to core training code required.
    *   `pages/_module_template/train.py` documents the full interface contract with inline comments covering both the text-native (embed file content directly) and media (embed generated description) strategies.
*   **Database Migration Fix:**
    *   Fixed a startup crash (`Error: Target database is not up to date`) in `app.py`. The migration sequence was inverted: `migrate()` (generate new revision) was running before `upgrade()` (apply pending revisions), which Alembic rejects. The order is now `upgrade()` first, then `migrate()`, ensuring pending migrations are always applied before any new drift is detected.
*   **Project Structure:**
    *   Files reshuffling for better project structure and maintainability. All core backend logic is now in `src/` folder.

### Version 0.3.5 (06.03.2026)
*   **OmniDescriptor Fixes:**
    *   `OmniDescriptor` no longer requires a workaround for streaming mode - the underlying bug was fixed by the MiniCPM-o-4.5 developers, so all extra workaround code has been removed.
    *   Model weights and code are now updated independently, making it possible to update the model code without re-downloading the large weight files.
*   **Universal Evaluator - Cross-Module Rating Model:**
    *   Introduced `UniversalEvaluator` (`pages/train/universal_train.py`) - a singleton `TransformerEvaluator` that trains on user-rated files from **all** media modules (music, images, videos, text) simultaneously, producing a single shared scoring model (`universal_evaluator.pt`).
    *   A two-phase embedding strategy is used during training: for text files, full chunked content is embedded via `TextSearch.process_files()` (cached, fast, identical to the old per-module text training path); for all other modules, `MetadataSearch.generate_full_description()` is used (filename + OmniDescriptor summary + internal metadata). The strategy for text files is controlled by `evaluator.text_embedding_method` in `config.yaml` (`"full_text"` / `"metadata"`).
    *   Added `generate_desc_if_not_in_cache` parameter to `MetadataSearch.generate_full_description()` (default `True`, preserving all existing callers). Training passes `False` to skip slow OmniDescriptor inference and rely only on already-cached descriptions.
    *   Replaced the "Train text evaluator" button on the training page with a "Train universal evaluator" button, wired through `pages/train/serve.py`.
*   **Videos Module - Model Ratings:**
    *   Wired `UniversalEvaluator` into `pages/videos/serve.py`. `update_model_ratings` now hashes video files, skips already-rated files (by `model_hash` comparison), generates metadata descriptions via `MetadataSearch`, and stores predictions in `VideosLibrary`. The `evaluator_hash` is passed to `get_files()` so stale ratings are detected and refreshed on browse.
*   **Text Module - Unified Evaluator:**
    *   Replaced the module-specific `TextEvaluator` with `UniversalEvaluator` in `pages/text/serve.py`. The text module now loads `universal_evaluator.pt` and uses the same embedding strategy as training (controlled by `evaluator.text_embedding_method`).
    *   `update_model_ratings` in the text module branches on the config switch: `"full_text"` uses `TextSearch.process_files()` (batch, cached); `"metadata"` uses `MetadataSearch.generate_full_description()` per file, consistent with all other modules.
    *   Removed the `TextEvaluator` singleton class from `pages/text/engine.py` and deleted the now-superseded `pages/text/train.py`.
*   **Bug Fixes:**
    *   Fixed a `UNIQUE constraint failed` crash in `update_model_ratings` for the Videos and Text modules. When duplicate files (same content, different paths) appeared in the media folder, both passed the already-rated filter, `query.filter_by(hash=...).first()` returned `None` for each, and both were queued for INSERT - triggering a hash collision on `bulk_save_objects`. New items are now deduplicated by hash before the bulk insert, consistent with the fix applied to the Images module in v0.2.15.

### Version 0.3.4 (27.02.2026)
*   **Model Re-Rating After Retraining:**
    *   Fixed a bug where `update_model_ratings` was never called after retraining an evaluator model. The `file_manager.get_files()` only passed files with `model_rating is None` to the rating callback, so files already rated by the old model were silently skipped. Added a new `evaluator_hash` parameter to `get_files()` - files whose `model_hash` differs from the current evaluator hash are now included in the re-rating list alongside unrated files. Applied consistently across all three modules (images, music, text).
*   **OmniDescriptor - New Omni-Modal Description Module (`src/omni_descriptor.py`):**
    *   Implemented `OmniDescriptor` - a singleton proxy class that manages a long-lived worker subprocess holding the [MiniCPM-o-4.5](https://huggingface.co/openbmb/MiniCPM-o-4_5) model. The subprocess isolates the CUDA context from the main Flask process and is automatically terminated after 5 minutes of inactivity (same pattern as `TextEmbedder`) to free GPU memory, then transparently restarted on the next call. 
    *   Model is loaded with BitsAndBytes 4-bit NF4 double-quantization (`bfloat16` compute dtype, `device_map='auto'`, `init_tts=False`) to fit within 8 GB VRAM. 
    * There are now several main methods for generating descriptions from different modalities available in the module:
        - **`describe_image(path, prompt)`** - generates a detailed text description of an image file via PIL + `model.chat()`.
        - **`describe_audio_sampled(path, n_samples, sample_duration_s, prompt)`** - memory-safe audio description. Samples `n_samples` evenly-spaced windows of `sample_duration_s` seconds each, runs the model independently on each, then synthesises a combined summary. Per-segment OOM is caught and skipped with `torch.cuda.empty_cache()`.
        - **`describe_video_sampled(path, n_samples, sample_duration_s, frames_per_segment, prompt)`** - memory-safe video description. For each of `n_samples` evenly-spaced time windows: extracts `frames_per_segment` frames via `cv2` (PIL images) and the corresponding audio waveform via `ffmpeg pipe:1 → io.BytesIO → librosa`. Passes both modalities together (`frames + [audio_chunk, prompt]`, `use_tts_template=True`) in a single `model.chat()` call per segment, then synthesises the per-segment descriptions into one summary. Uses `num_beams=1` throughout to prevent beam-search OOM.
        - **`describe_text(text, prompt)`** - summarises long text via `model.chat()`.
    *   Added a comprehensive `__main__` test suite to ensure its stability and reliability of the new module.
    *   To make all the modalities properly working, new set of liraries added to the `requierments.txt` (`opencv-python-headless`, `minicpmo`, `hyperpyyaml`, `diffusers`, and `onnxruntime`).
*   **UX Improvements:**
    *   When gathering descriptions and text embeddings in the `metadata-based-search` mode, the status message reflects the amount of files processed out of the total and estimated time remaining, providing better feedback during long operations.
    
### Version 0.3.3 (17.02.2026)
*   **Text Module - Evaluator Model & Training:**
    *   Implemented a new `TransformerEvaluator` architecture in `src/scoring_models.py` - a lightweight transformer-based model that takes a variable-length sequence of chunk embeddings as input and produces a rating score. Architecture: linear projection (1024→1024 for now, reserve for future optimizations), learnable [CLS] token, sinusoidal positional encoding, 2-layer TransformerEncoder (4 heads, FFN=512), LayerNorm + MLP head → 11-class rating output. Designed to train efficiently on 8GB GPUs.
    *   `TextEvaluator` in `pages/text/engine.py` now inherits from `TransformerEvaluator` instead of the MLP-based `Evaluator`, properly handling the variable-length chunk embeddings that text files produce.
    *   Created `pages/text/train.py` with `train_text_evaluator()` function, analogous to the existing image evaluator training pipeline. Queries rated text files from the database, generates embeddings via `TextSearch`, and trains the transformer model with progress reporting.
    *   Wired up `text_evaluator.predict()` in `pages/text/serve.py` - model ratings for text files are now computed and saved to the database instead of being hardcoded to `None`.
*   **Text Module - User Rating UI:**
    *   Added an interactive star rating widget (using the shared `StarRatingComponent`) to the text file preview modal. Users can now rate text files directly from the viewer.
    *   Added `emit_text_page_set_text_rating` socket event handler in `pages/text/serve.py` to persist user ratings to the `TextLibrary` database.
*   **Training Page:**
    *   Added "Train text evaluator" button to the training page (`pages/train/page.html`), with full backend wiring in `pages/train/serve.py`. The button is disabled during any active training session, consistent with existing music and image evaluator buttons.

### Version 0.3.2 (09.02.2026)
*   **Configuration & Deployment (Breaking Changes):**
    *   **Replaced `.env`-based configuration with `docker-compose.override.yaml`.** Users now configure all instance-specific settings (media folders, port, container name, authentication) in a single `docker-compose.override.yaml` file instead of a separate `.env` file. A well-documented `docker-compose.override.example.yaml` template is provided. This is a **breaking change** - existing users must migrate their settings from `.env` to the new override file (see migration notes below).
    *   **Multiple media folders per module.** Each module (images, music, text, videos) now supports mounting multiple source folders. Each folder appears as a separate top-level directory in the app's file browser. All search, sorting, and recommendation features work seamlessly across all folders. No application code changes were needed - this works via Docker's native multi-mount capability.
    *   **Multi-instance support via `instances/` directory.** Running multiple Anagnorisis instances simultaneously (e.g. for different family members) is now cleanly supported using separate compose override files in the `instances/` folder, with example templates provided.
    *   **`docker-compose.yaml` is now developer-maintained only.** Users should not edit `docker-compose.yaml` directly. All user-specific configuration goes into `docker-compose.override.yaml` (for single instance) or files in `instances/` (for multiple instances).
    *   The `.env` file and `.env.example` are no longer used by the project. Environment variables previously set in `.env` (such as `CONTAINER_NAME`, `ANAGNORISIS_USERNAME`, `ANAGNORISIS_PASSWORD`, `ALLOW_FILE_DELETION`) are now set in the `environment` section of the override file.
*   **Migration from previous versions:**
    1.  Create your new config: `cp docker-compose.override.example.yaml docker-compose.override.yaml`
    2.  Move your folder paths from `.env` into the `volumes` section of `docker-compose.override.yaml`.
    3.  Move any environment variables (`ANAGNORISIS_USERNAME`, `ANAGNORISIS_PASSWORD`, `ALLOW_FILE_DELETION`) into the `environment` section.
    4.  Your `.env` file can be deleted after migration.
    5.  Start with `docker compose up -d` as before.

### Version 0.3.1 (06.02.2026)
*   **Images Module:**
    *   Fixed a crucial issue that prevented recommendation model training process initialization.
    *   Added `Show full search description` option to the context menu. This opens a `MetaEditor` modal showing the full text payload used for generating search embeddings, with all necessary backend functionality implemented to support this feature.

### Version 0.3.0 (19.01.2026)
*   **File Management & Database:**
    *   Moving files between folders is now fully supported in the `Images` module. When a file is moved, its database entry is updated to reflect the new path, ensuring that all associated metadata and embeddings remain intact. This prevents duplication of entries and maintains data integrity. Both full destination paths on the host system and Docker container paths are correctly handled. External meta files (`{file}.meta`) are also moved accordingly as well.
    *   Fixed an issue in `FileManager` where old data might be returned when files were moved between folders. The file list is now properly refreshed after such operations to ensure accurate display of current files and folder structure.
*   **Module Initialization & Loading:**
    *   Current initialization status of all modules is now sent to the client at each `socketio` connection. This ensures that if the client refreshes the page during initialization, it will still receive accurate status updates.
*   **System & Architecture:**
    *   Fixed an issue in `TextEmbedder` where process termination could lead to model downloading process being interrupted, resulting in model corruption. The model downloading process is now properly handled and no process is terminated until crucial operations are completed.
*   **Testing & Quality Assurance:**
    *   Thoroughly tested the application from complete fresh install with full initialization and usage of all modules to ensure stability and correctness of the initialization processes.

### Version 0.2.15 (15.01.2026)
*   **Architecture & Process Management:**
    *   `TextEmbedder` is now spawned in a separate process to avoid GPU memory cleanup issues. This simplifies the codebase by eliminating the need for `ModelManager` to track models loaded in CPU/GPU memory. After the idle timeout expires, the entire process is terminated, freeing all resources automatically.
*   **Module Initialization & Loading:**
    *   Module initialization now provides loading status feedback to the UI. A special loading page is displayed while the initialization process is running, showing appropriate status messages until the module is fully ready.
*   **Music Module:**
    *   Fixed an issue where errors during song details fetching would clear the entire playlist and reload the page. Songs with errors are now skipped without penalizing their recommendation score, preventing the playlist from being blocked by a single problematic file.
*   **Images Module:**
    *   Fixed an issue where search status messages were not being displayed in the UI.
    *   Fixed a `UNIQUE constraint` error that could occur when updating model ratings if duplicate files were present in the media folder. Duplicates are now properly handled during the rating update process.

### Version 0.2.14 (30.11.2025)
*   **Search & Metadata:**
    *   Implemented a smooth maximum formula for `TextSearch` similarity calculations. This prioritizes files with multiple relevant chunks over those with a single high-scoring chunk, improving search relevance.
    *   Expanded metadata-based search to include internal file metadata (e.g., band name, release year for `Music`) as well as `.meta` files information, providing richer context beyond just file names.
    *   Refined `filter_by_text` in `CommonFilters` to prioritize exact matches during fuzzy matching, resulting in more accurate file name searches.
*   **Music Module:**
    *   Fixed a critical sorting issue where `files_list` and `files_data` misalignment caused inconsistent recommendation scores. File order is now guaranteed, ensuring recommendations work as expected.
    *   Added a `Show full search description` option to the context menu. This opens a `MetaEditor` modal showing the full text payload used for generating search embeddings.
*   **System & Architecture:**
    *   Enhanced `base_search_engine.py` with robust model loading logic. It now detects corruption during model downloads and automatically triggers a re-download if necessary.
    *   Critical directories (`logs`, `database`, `models`, `cache`) are now automatically created within `{PROJECT_CONFIG_FOLDER_PATH}` on startup, preventing crash loops on fresh installs.
    *   Improved path traversal prevention to correctly handle and display valid filenames containing `..` (e.g., `Song..mp3`) while still blocking malicious paths.
*   **UI & Components:**
    *   Updated `MetaEditor.js` to support a read-only mode, allowing it to be used for viewing informational text (like search descriptions) without editing capabilities.
*   **Documentation:**
    *   Updated `README.md` with a troubleshooting section addressing common Docker mount permission issues.
*   **Maintenance:**
    *   Minor code refactoring and cleanup.

### Version 0.2.13 (03.11.2025)
*   **UI & Components:**
    *   Added a reusable context menu component (`pages/ContextMenuComponent.js`) to simplify building right‑click menus throughout the UI.
    *   Added a reusable `MetaEditor` (`pages/MetaEditor.js`) for viewing/editing `.meta` files associated with media files.
    *   File tiles now show a bottom‑left “.meta” icon when a corresponding `{file}.meta` exists; clicking it opens the Meta Editor. Implemented in `Images`, `Music`, and `Videos` modules.
*   **Search & Actions:**
    *   Added `Find similar images` and `Find similar music` actions to the respective module context menus to start similarity searches from a selected file.
    *   Fixed a sorting issue where similarity results were reversed (closest items appearing last); order is now correct.
*   **Hashing & Database:**
    *   Hash algorithm type and version are now stored in the database alongside file hashes for traceability and future migrations. Note: a fail‑safe mechanism for hash algorithm changes is not implemented yet and is planned for later work.
*   **Socket Messaging & Concurrency:**
    *   Fixed a threading lock/race condition in `CommonSocketEvents` that could drop some of the last messages; messaging is now reliable.
*   **Music Module:**
    *   Internal metadata gathering now downsizes artwork images automatically to reduce cache storage footprint.
    *   Improved embeddings gathering flow for better speed and reliability.
*   **Status & Logging:**
    *   Added more granular status messages during hash and embedding gathering to better reflect ongoing work in the UI.
    *   Cache logging now appears only when new information is available (reduced noise).
*   **GPU/Model Management:**
    *   Gradual improvements to GPU/CPU memory handling in `src/model_manager.py` and related code. Models are no longer kept on GPU by default after startup; they load/unload dynamically as needed.
*   **Documentation:**
    *   README updates and small cleanup.

### Version 0.2.12 (27.10.2025)
*   **Database & Path Sync:**
    *   Paths in the DB are now automatically kept in sync when a file with a known hash is encountered (to have a relevant path to the file if it was moved/renamed). This ensures stored relative paths remain valid without manual reindexing.
*   **Socket Messaging & UX:**
    *   Centralized status throttling in `CommonSocketEvents` (pages/socket_events.py) with a per-sid 0.25s interval and “last message wins” guarantee. Prevents UI floods and removes the need for ad‑hoc throttling across modules. All modules using socket events have been updated accordingly.
*   **Training:**
    *   Training of recommendation models in both `Music` and `Images` modules is fixed and operates as expected.
    *   Global in‑memory training lock added: when training is active, start buttons on the Train page are disabled to prevent concurrent runs. State survives page refresh within the same server process.
*   **Hashing & File Operations:**
    *   Implemented a fast `Videos` module hashing scheme that reads large files partially and uses `xxh3` for significant speedups.
    *   Improved status tracking for long operations (hash gathering, embedding extraction) and fixed several issues in file management and hash collection flows.
*   **Maintenance:**
    *   Minor code refactoring and cleanup.

### Version 0.2.11 (25.10.2025)
*   **Caching & Storage:**
    *   Introduced a new, more sophisticated `TwoLevelCache` (`src/caching.py`) that exposes only `get`/`set`, manages RAM/disk tiers internally, defers disk writes (batch flush ~every 5 minutes or on thresholds), and shards key/value files by key hash to reduce I/O. It tries to minimize disk reads and writes. To make reads faster sharding approach is used where (key,values) pairs are stored in multiple files instead of a single big file depending on the key's hash.
    *   Replaced all legacy caching mechanisms across the project with `TwoLevelCache` instances.
    *   Moved all file embeddings out of the database and into the cache, significantly reducing DB size while preserving performance due to pinpoint-fast cache reads.
*   **Embedding Extraction:**
    *   Simplified extraction procedures for maintainability.
*   **Search & Metadata:**
    *   `src/metadata_search.py` now incorporates the contents of `{file_name}.meta` files into metadata search for better results.
*   **GPU & Model Management:**
    *   Improved GPU memory handling in `src/model_manager.py` for dynamic load/unload of multiple models and fixed several related memory issues.
*   **Maintenance:**
    *   General code cleanup and refactoring.
*   **Known Issues:**
    *   Training of recommendation models is temporarily broken and will be fixed in upcoming versions.

### Version 0.2.10 (19.10.2025)
*   **Architecture & Search Enhancements:**
    *   Text-based filtering in `common_filters.py` now supports three modes: `file-name`, `semantic-content`, and `semantic-metadata`. Metadata-based search is currently partial (file names and paths only). It will be extended to include richer file's metadata and `{file_name}.meta` contents, using `TextEmbedder` as the backbone.
    *   Adopted `rapidfuzz` for fast and accurate file-name search; added to `requirements.txt`.
    *   All filters now return scores (instead of sorted lists). Sorting order and temperature-based sampling are now applied centrally in `file_manager.py -> get_files`, which has been updated to work with `mode`, `order`, and `temperature` parameters.
*   **UI & Componentization:**
    *   Added a reusable `SearchBarComponent.js` to provide an advanced search bar with mode/order/temperature controls that can be included across modules.
    *   `get_files` now accepts `temperature` directly as a parameter (replacing terminal-style flags), so users can adjust it in the UI.
    *   All modules have been adapted to work with the improved search bar and filtering flow.
*   **Maintenance:**
    *   Minor code refactoring.

### Version 0.2.9 (06.10.2025)
*   **Architecture & Search Enhancements:**
    *   Refactored text embedding processing into a dedicated `TextEmbedder` class (`src/text_embedder.py`). This improves code organization, maintainability, and allows the module to be reused for metadata-based search across the application.
    *   Implemented a new metadata-based search feature across the `Images`, `Music`, and `Text` modules. This search currently uses file names and paths, with plans to incorporate more detailed metadata from file contents and `.meta` files in the future.
    *   The metadata search is powered by a new, more sophisticated two-level caching mechanism (`src/metadata_search.py`) featuring both RAM and persistent disk storage. If this proves effective, it will be adopted project-wide.
    *   Modified the `compare` method in `base_search_engine.py` to return pure cosine similarity values (in the -1 to 1 range) by removing the final logit scaling step. This ensures a more accurate and interpretable similarity score, especially when combining content and metadata embeddings.
*   **UX & Debugging Improvements:**
    *   Added a real-time log viewer, accessible via a ">_" button in the header. This feature uses a new `src/log_streamer.py` module with `watchdog` and WebSockets to stream log file changes directly to the UI for easier development and debugging.
    *   Search results across all modules now display a breakdown of similarity scores (semantic, metadata, and total), providing better insight into why specific files are returned.
    *   The `EmbeddingGatheringCallback` now shows the name of the current process (e.g., "metadata"), giving the user clearer feedback on background activities.
*   **Testing & Deployment:**
    *   Created a new lightweight Dockerfile and `docker-compose.yaml` in the `tests` folder to facilitate isolated testing of encapsulated features like `src.text_embedder` and `pages.text.engine`.
*   **General Maintenance:**
    *   Performed minor code cleanup and refactoring across the project.

### Version 0.2.8 (30.09.2025)
*   **Recommendation & Search Enhancements:**
    *   Introduced a `temperature` parameter to the `sort_files_by_recommendation` method in `pages/recommendation_engine.py`. This allows for dynamic control over the randomness of recommendations, where `0` provides a strict, score-based order and higher values increase randomness.
    *   Implemented support for terminal-style arguments (`-t` or `--temperature`) in the `Music` module's search bar for the `recommendation` filter, allowing users to adjust the sorting temperature directly. For now it is only takes into account the file name and relative path, while in the future data from the `{file_name}.meta` files will be used as well. I also want to add this functionality to all other modules as well, working in the same vein. This will complete the main search functionality.
    *   Enhanced the `Text` module's search capabilities by processing file metadata (file name and relative path) as a separate embedding. This provides more contextually aware and intuitive search results. Relative scores (total/content_score/meta_score) are also displayed in the search results now.
*   **Robustness & UX Improvements:**
    *   Improved the file hash gathering process to save progress to the cache every 60 seconds, preventing data loss during long operations.
    *   Enhanced the status display for hash gathering to show percentage completion for better user feedback.
    *   Standardized the `file size` filter to `file_size` in the `Music` and `Images` modules for consistency with the new terminal-style command parsing.
*   **Bug Fixes:**
    *   Resolved a critical `TypeError` by fixing a data serialization issue that occurred during search operations in the `Music` and `Images` modules, ensuring NumPy data types are correctly converted before being sent to the client.

### Version 0.2.7 (18.09.2025)
*   **Architecture & Code Refactoring:**
    *   Global code refactoring in the `serve.py` files across all modules (`Images`, `Music`, `Text`, `Videos`) to significantly reduce code duplication and improve maintainability.
    *   Moved common `get_files` functionality into the `FileManager` class (`pages/file_manager.py`) for reuse across all modules.
    *   Centralized common filtering options into a new `CommonFilters` class (`pages/common_filters.py`), while retaining domain-specific filters in their respective modules.
    *   Pinpoint changes in some elements names and other minor changes.
*   **UI & UX Enhancements:**
    *   `FileGridComponent` is now consistently used in all modules (`Images`, `Music`, `Text`, and `Videos`) for displaying files, replacing custom implementations.
    *   The columns in `FileGridComponent` now dynamically adjust to the screen size, with a minimum tile width of `18rem`.
    *   `Update Playlist` buttons are now correctly disabled until file data has completely loaded, preventing potential errors.
    *   The displays for media folder paths have been temporarily hidden, as these paths are now statically configured in the `.env` file.
*   **Video Module Improvements:**
    *   Implemented autoplay for the next video in the `Videos` module. When the current video finishes, the next one in the current view will start playing automatically.

### Version 0.2.6 (31.08.2025)
*   **Security & Path Traversal Prevention:**
    *   Implemented comprehensive path traversal attack prevention using a minimal `@app.before_request` decorator that automatically protects all routes from dangerous patterns like `..`, `%2e%2e`, `/etc/`, and other traversal attempts.
    *   Enhanced Docker security configuration with `security_opt` to reduce attack surface for potential self-hosting scenarios.
*   **Performance & Device Management:**
    *   Fixed critical issue with slow embedding processing caused by device mismatch between models and tensors, which was forcing CPU processing instead of GPU acceleration. Embedding generation now correctly utilizes GPU resources.
*   **Text Module Enhancements:**
    *   Added Markdown and HTML preview functionality to the Text module, allowing users to view `.md` files as rendered content alongside the raw text view.
    *   Text file viewer now includes tabs for Raw, Markdown, and HTML views in the modal interface.
*   **Audio Format Support:**
    *   Added `.opus` file format support to the Music module's supported formats in `config.yaml`.
*   **Architecture & Deployment:**
    *   Streamlined project startup by removing `run.sh` and `run.bat` scripts. All necessary commands are now integrated into the `Dockerfile`, with local environment execution temporarily unsupported in favor of containerized deployment.
    *   Improved Docker container configuration for enhanced security and better isolation in preparation for self-hosting deployment scenarios.
*   **Folder & File Management:**
    * Some preparation work for future folder and files management in `FolderViewComponent` has been made.

### Version 0.2.5 (25.08.2025)
*   **Docker & Environment Enhancements:**
    *   Fixed an issue where the virtual environment created by Docker was overwritten when source code was mounted into the container. The virtual environment is now created in a separate location (`/venv`) to avoid conflicts with code mounting.
    *   Resolved constant login requirements even when `ANAGNORISIS_USERNAME` and `ANAGNORISIS_PASSWORD` environment variables are not set. The application now runs without authentication when these variables are not configured.
    *   Implemented new container setup workflow based on `.env` file parameters for easier configuration management. Container logs are now stored as `{CONTAINER_NAME}_log.txt` by default instead of `container_log.txt`.
*   **Configuration & File Management:**
    *   Replaced `PROJECT_CONFIG_FOLDER_NAME` with `PROJECT_CONFIG_FOLDER_PATH`, which now contains the full path to the directory where personal database and trained recommendation models are stored.
    *   Separated model storage into two distinct locations: all embedding models are stored in `./models/` in the project root, while personal trained models are stored in `{PROJECT_CONFIG_FOLDER_PATH}/models/`. This prevents redundant model downloads when running multiple project instances on the same host.
    *   `config.yaml` file is no longer created in the `{PROJECT_CONFIG_FOLDER_PATH}` directory, as it is no longer necessary.
*   **Module Improvements:**
    *   Added seed-based sorting to `Music` and `Images` modules, matching the functionality already present in the `Videos` module. This ensures consistent file order when navigating between pages.
*   **Dependencies:**
    *   Added missing `moviepy` dependency to `requirements.txt` to ensure video processing functionality works correctly.
    *   Added `protobuf` library to `requirements.txt` as required by the `transformers` library for model processing.

### Version 0.2.4 (29.06.2025)
*   **Security Enhancements:**
    *   Authentication with `ANAGNORISIS_USERNAME` and `ANAGNORISIS_PASSWORD` environment variables for basic privacy has been implemented. If these variables are not set, the application will run without authentication. This provides a layer of security when the application is exposed to the internet.
    *   Very basic prevention of `Directory Traversal` attack has been implemented in `serve.py -> get_files` methods of all modules (`Images`, `Music`, `Text`, `Videos`). Further work is needed in file sharing methods to ensure no sensitive files can be accessed when the application instance is exposed over a network.
*   **Module Improvements:**
    *   Added external metadata (stored in `{file_name}.meta` file) viewing and editing functionality in the `Images` module. This feature lays groundwork for future additional text-based image embeddings that will enhance search, filtering, and recommendation capabilities.
    *   Implemented seed-based sorting for the `Videos` module. This feature ensures consistent file order in the preview when navigating between pages and is a potential approach for other modules. If it proves effective, similar functionality will be applied to the `Images`, `Music` and `Videos` modules.
*   **Configuration & Minor Enhancements:**
    *   Expanded supported media formats in `config.yaml` to include `.wav`, `.ogg`, `.aac`, `.m4a` for music; `.htm`, `.html` for text; and `.mpeg`, `.mpg` for videos.
    *   Minor internal code cleanup, including removal of commented test sections in `engine.py` files and adjustment of button IDs in the `videos` module frontend.
    *   The `restart: unless-stopped` option has been commented out in `docker-compose.yaml` to allow for easier manual stopping of the container during development.
 

### Version 0.2.3 (09.06.2025)
*   **Core Architecture Enhancements:**
    *   Refactored the `ImageSearch` (`pages/images/engine.py`) and `MusicSearch` (`pages/music/engine.py`) classes to inherit from the `BaseSearchEngine`, following the pattern established in the `TextSearch` class. This further reduces code duplication and enhances maintainability. To test the correctness of the code, test cases were created for both engines, which can be run with the command:
        *   `python3 -m pages.images.engine`  
        *   `python3 -m pages.music.engine`
*   **Video Module Improvements:**
    *   Implemented 'recommendation' based sorting for videos, leveraging the existing recommendation logic and preparing the `VideosLibrary` database model to track user ratings, model ratings, play/skip counts, and `last_played` timestamps. For now only the `last_played` timestamp is used for the recommendation, but the rest of the fields are prepared for future use.
    *   Fixed a critical issue where video transcoding might get stuck, preventing the video from being played because incorrect error handling was implemented. Now the transcoding process is properly monitored, and if it fails, the logs get information about it yet the transcoding process continues.
    *   Fixed an issue when videos might get started from 4-5 seconds mark instead of the beginning. Now videos always starts from the beginning as expected.
    *   Updated the video module frontend (`pages/videos/js/main.js`, `pages/videos/page.html`) to utilize the shared `FileGridComponent` and `FolderViewComponent` for a consistent and robust user interface.
*   **Minor Fixes & Enhancements:**
    *   The database export functionality (`/export_database_csv`) now correctly excludes `chunk_embeddings` (from the Text module) to prevent issues with large binary data during export.
    *   User ratings in the `Images` module are now consistently stored as floating-point numbers in the database, allowing for more precise user input (e.g., 8.5 out of 10).
    *   `config.yaml` has been updated for consistency and clarity, explicitly adding `embedding_model: null` for the `videos` section as no embedding model is currently used for videos.

### Version 0.2.2 (04.06.2025)
*   **Core Architecture Refactoring:**
    *   Introduced a `BaseSearchEngine` abstract class (`src/base_search_engine.py`) to encapsulate common functionality for embedding models (initialization, model downloading, caching, file processing flow).
    *   Refactored the `TextSearch` (`pages/text/engine.py`) class to inherit from the `BaseSearchEngine`, eliminating significant code duplication and improving code structure and maintainability. For each engine test cases were also created to ensure the correctness of the code. Test might be run with these commands:
        *   `python3 -m pages.text.engine`  
    ( All other engines, such as `ImageSearch` (`pages/images/engine.py`) and `MusicSearch` (`pages/music/engine.py`) are yet to be refactored. )
    *   Data for testing `TextSearch`, `ImageSearch`, and `MusicSearch` engines has been added to the respective `engine_test_data` folders.
*   **Improved Model Management:**
    *   Integrated the `ModelManager` into the `Evaluator` classes (`src/scoring_models.py`) and the refactored Search Engines (via the `BaseSearchEngine`), enabling lazy loading and automatic unloading of models from GPU memory when idle, leading to more efficient resource usage.
*   **Configuration & Minor Fixes:**
    *   Ensured the `embedding_model` configuration setting is consistently present in `config.yaml` for `music` and `images` modules.
    *   Corrected a minor typo in the music module's JavaScript (`#seach_button` to `#search_button`).
    *   `Dockerfile` configuration has been updated to use the latest `jinaai/jina-embeddings-v3` model (it temporally uses non-local files, with `local_files_only=False, trust_remote_code=True` as I cannot yet find a way to run the updated model without it).

### Version 0.2.1 (29.04.2025)
*   **Docker Enhancements & Security:**
    *   Changed default Docker port mapping (`docker-compose.yaml`) to bind to `127.0.0.1` (localhost) instead of all interfaces (`0.0.0.0`), enhancing default security by preventing accidental exposure on the local network. Users wanting broader network access will need to modify the `docker-compose.yaml` file.
    *   Made the container log file name dynamic in the `Dockerfile` (using `CONTAINER_NAME` environment variable, defaulting to `container_log.txt`), allowing for easier identification if running multiple instances. Corresponding changes made in `.gitignore`.
*   **Configuration Flexibility:**
    *   Introduced `PROJECT_CONFIG_FOLDER_NAME` environment variable (defaulting to `Anagnorisis-app`) to allow customization of the sub-directory within the main `DATA_PATH` where project-specific data (database, models, config) is stored (`app.py`).
    *   Removed now-redundant path configurations (`database_path`, `migrations_path`, `models_path`, `cache_path`) from the default `config.yaml` as they are derived programmatically in `app.py`.
    *   Changed default `media_directory` values in `config.yaml` from empty strings (`''`) to `null` for clearer indication when a path is not set.
*   **Text Module Major Update:**
    *   Integrated the `TextSearch` engine (`pages/text/engine.py`) for embedding generation using the configured `embedding_model` (e.g., `jinaai/jina-embeddings-v3`).
    *   Implemented backend logic (`pages/text/serve.py`) for semantic search based on text queries and file content embeddings.
    *   Added database model (`pages/text/db_models.py`) for storing text file information and chunk embeddings.
    *   Significantly updated the UI (`pages/text/page.html`, `pages/text/js/main.js`):
        *   Added folder view navigation.
        *   Implemented file grid display using `FileGridComponent`.
        *   Added search bar functionality.
        *   Integrated pagination using `PaginationComponent`.
        *   Improved modal viewer with tabs (Raw, Markdown, HTML) and better layout.
        *   Added logic for saving edited text content.
        *   Added controls for setting the media path.
*   **Video Module Streaming:**
    *   Implemented experimental **HLS (HTTP Live Streaming)** support for video playback (`pages/videos/serve.py`, `pages/videos/js/main.js`, `pages/videos/page.html`).
    *   Uses `ffmpeg` on the server for **on-the-fly transcoding** of video files into HLS format (.m3u8 playlist and .ts segments).
    *   Added backend logic to manage transcoding processes and serve HLS files.
    *   Integrated `hls.js` on the frontend for playing HLS streams in the video modal.
    *   Includes basic stream management (start/stop) and automatic cleanup of old transcoding processes.
*   **Module Robustness:**
    *   Added checks in `Images`, `Music`, `Text`, and `Video` backend (`serve.py`) to handle cases where the respective `media_directory` is not set (`null` in config), preventing errors and showing appropriate status messages.
*   **Dependencies:**
    *   Added `sentence-transformers` and `einops` to `requirements.txt`.
    *   Specified `numpy<2` in `requirements.txt` for compatibility with `jinaai/jina-embeddings-v3` model.

### Version 0.2.0 (04.04.2025)
* Docker container for running the project configured and extensively tested to make running the project easier.
* Database is now automatically created if it doesn't exist, and migrations are applied on application startup, removing the need for manual `flask db` commands.
* Introduced `DATA_PATH` environment variable and `--data-folder` argument to manage all user-specific data (database, models, config as well as user's personal data) in a central location (defaults to `./project_data`).
* Implemented a system where user settings in `{DATA_PATH}/Anagnorisis-app/config.yaml` override the base `config.yaml` for proper configuration personalization that do not get overwritten after the `git pull`. The user config is created automatically from the default if it doesn't exist.
* Required embedding models (`SigLIP`, `CLAP`) are now downloaded automatically on first use if not found in the configured `models_path`.
* Revised `README.md` with updated setup instructions (including Docker) and adjusted `run.sh`/`run.bat` for the new data folder structure and automatic migrations.
* Changed default host to `0.0.0.0` for it to be compatible with Docker. 
* Added `huggingface-hub` to `requirements.txt` (necessary for automatic models download).

### Version 0.1.7 (29.03.2025)
* Added a Docker-based test (`tests/test_docker_build.sh`) to automatically verify project installation and startup in a clean environment.
* Removed the old `_music_v0.1.1` module code, outdated research folders, and documentation. Removed the `TTS` dependency from `requirements.txt`.
* Added initial `text` configuration section to `config.yaml`.
* Created a reusable `FileGridComponent.js` for displaying file grids and integrated it into the 'Images', 'Music', and 'Text' modules for consistency.
* Created a reusable `PaginationComponent.js` for pagination and integrated it into the 'Music', and 'Text' modules.
* Method `show_search_status` separated into a `CommonSocketEvents` and now targeted for each folder separately. This approach will later be used for almost all socket events.
* In `Music` module the playlist now is automatically cleaned up in case any error occurs during the playback. This should prevent the playlist being not accessible after the error.
*   **New 'Text' Module:** Introduced a new 'Text' module for viewing and editing `.txt` and `.md` files:
    *   Includes folder navigation and file grid display.
    *   Features a modal window for viewing content with Raw/Markdown/HTML tabs (Markdown/HTML rendering not yet implemented).
    *   Allows basic text editing and saving.

### Version 0.1.6 (01.03.2025)
* Implemented control of music playback (play/pause, next/previous track) through browser's `navigator.mediaSession` API. This allows control from external devices and browser UI elements that support media sessions (like media keys on keyboards, lock screen controls, and some bluetooth devices).
* Fixed issues that prevented `full_play_count` and `skip_count` from being correctly incremented and stored in the database when songs are played or skipped. These counts are now accurately tracked and persisted for use in the recommendation algorithm.
* Implemented saving and restoring of the music module's state using `localStorage`.  The current playlist, current song, playback position, and play/pause status are now preserved across page refreshes, providing a more seamless user experience.
* Resolved an issue that prevented searching for music files by their full file path within the `Music` module's library view. Searching by file path now functions as expected.
* Re-enabled and fixed the 'Find similar' button in the `Music` module. This feature now correctly initiates a similarity-based search for music tracks.
* Improved playlist behavior: When playing music from the library view, if files are selected, only the selected files are added to the playlist and played. If no files are selected, all files in the current view are played (as before).
* Added a `Last played` column to the music library grid view in the `Music` module. This column displays the last time each song was played, providing users with playback history information directly in the library view.
* `Open` button in the music library grid view in the `Music` module now works correctly. This button allows users to open the folder containing the selected music file directly from the library view.

### Version 0.1.5 (24.02.2025)
* Added metadata caching mechanism and integrated in both `Images` and `Music` modules. New `CachedMetadata` class in `pages/file_manager.py` to handle caching of file metadata, improving performance by reducing redundant metadata extraction. Metadata is cached using file hashes as keys, making the cache robust to file renaming and moving. Cache entries are automatically invalidated after three months to ensure data freshness.
* Now process of gathering metadata is properly displayed in the status bar in the `Images` and `Music` modules displaying the current percent and total number of files to process.
* Limitations on `resolution` and `proportion` search in the `Images` module have been removed, allowing users to search for images of any resolution or proportion.
* Model training for `Music` module has been properly restored and tested.
* While training the metric has been updated from `1 - MAPE` to `1 / (1 + MAPE)` as it better reflects the quality of the model and cannot give negative values.
* Fixed an issue with sorting files in the `Music` module.

### Version 0.1.4 (16.02.2025)
* `ask.py` script and related functionality is now moved to a separate project called [InsightCoder](https://github.com/volotat/InsightCoder).
* `Recommendation engine` is now separated into its own module and supports creating a list of recommendation based on the same algorithm as it was used in the radio mode. It supposed that the same module would be used for all other types of data supported by the project in the future.
* `Star rating` module is now supports fractional values and is more responsive to the user input. Database entries related to the user's ratings are now also stored as floats.
* `Music` module are now supports filtering files by `rating`, `similarity`, `file size`, `random` and `recommendation`.
* Returned the ability to rate songs in the `Music` module. Play and skip counts are yet to be implemented.
* Number of files per page in the `Music` are set to 30 as aquering metadata for music files is quite slow for now.
* Some primitive experiments with p2p connection were done in the `Research` folder.
* Improved extension loading mechanism using importlib.import_module for better code organization and maintainability.
* Added configurable database path via database_path setting in `config.yaml`.
* Minor UI improvements in the `Music` module

### Version 0.1.3 (04.02.2025)
*   The project now checks for the existence of the image/music evaluator models before attempting to load them. If they don't exist, a dummy model is created to prevent errors. This ensures the application can start even if the evaluator models haven't been trained yet.
*   Replaced the old audio embedding model `m-a-p/MERT-v1-95M` with `LAION-AI/CLAP`. New audio embedding model is based on the CLIP architecture that allows to make text search over the local music library.
*   Significantly refactored the music embedder code. Removed the `AudioEmbedder` class from the main code. Implemented a new `MusicSearch` class in the `Music` page module, analogous to the `ImageSearch` class in the `Images` module, to handle music library searches.
*   The `pages/music/serve.py` file was updated to reflect the changes in the way music is processed and embeddings are handled.
*   When importing the database from a `.csv` file, empty fields are now correctly handled as `None` instead of empty string that caused errors.
*   Some more info has been added to `project_info` folder to better handle AI interactions.
*   The `ask.py` script can now use prompts from `.txt` files, allowing for more complex and reusable queries.
*   Updated the `requirements.txt` file to include the sentencepiece library.
*   Restored song metadata representation in the song's control panel. It now correctly displays album's image, artist, album, and title as well as current rating of the song. Changing the song's metadata and rating is not yet available.

### Version 0.1.2 (28.01.2025)
*   File `requirements.txt` has been updated to include all necessary packages for the project without particular versions.
*   Roadmap of the project has been added to the wiki.
*   A complete overhaul of the 'music' module has been done. The old version was moved to `pages/_music_v0.1.1`. The new module is more similar to the 'images' module, using a grid-based layout for music files and folders, and has many of its improvements, such as file-hashing mechanism for faster file access, and a similar approach to the database interaction. Some of the features from the old 'music' module are still missing, but they will be added in the future. Note, that recommendations are not currently working as the recommendation model is not yet utilized. Embedding model is also going to be changed, so music embeddings are not yet in use. There is no way to rate the music for now either. The old `js/radio.js` file was removed, and all the radio functionality was moved into the new `js/main.js`. Also `js/library.js` was removed and its functionality were also added to the `js/main.js`. The `pages/music/data` folder was also removed, and the background sound file was moved to `static/background-music`.
*   The music DB model (`MusicLibrary`) has been changed to resemble the images DB model. It now includes `id`, `hash`, `file_path`, `user_rating`, `user_rating_date`, `model_rating`, `embedding`, `embedder_hash`, `full_play_count`, `skip_count`, and `last_played` fields. Many of the old fields are not used anymore. Embeddings are not yet stored in the DB.
*   The `ImagesLibrary` model now also includes `model_hash` and `embedder_hash` fields. The model rating is also now a float number for images.
*   Added a separate folder `project_info` with general data about the project and its structure, and `ask.py` script that uses "Gemini 2.0 Flash-exp" (over 1m tokens of context) to answer questions about the project and help in its development. The `ask.py` sends the entire codebase to Google's servers, so users should avoid including any sensitive data in the project. The response of the model is saved into `project_info/llm_response.md` file for review. The user prompt is stored to the `project_info/llm_prompt.txt` file for review.
*   Whole database now could be downloaded as a `.csv` file via the download button on the navbar.
*   Added import into the database from a `.csv` file using the upload button on the navbar. The import only adds new data and does not remove any data that already in the DB.
*   The project now uses a file list caching mechanism (`CachedFileList`) and a file hash caching mechanism (`CachedFileHash`) in `pages/file_manager.py`, for better performance of file management. The old way of database access was removed. The new one uses the same approach as in 'images' module.
*   The `pages/utils.py` file was created and the methods `convert_size` and `convert_length` were moved to it.
*   The rating components are now a separate .js file `pages/StarRating.js` that is used in both 'images' and 'music' modules.
*   The database now stores the embedding model's hash (`model_hash`) to prevent using old rating models.
*   The project is now more strictly divided by extensions, as all of the data related to an extension, including the DB models, server-side python code, html page and front-end javascript code is now stored in a separate folder. The base code now acts as a base for all the extensions. The extensions are automatically discovered.
*   The navbar now contains a button to download the whole DB as a .csv file.
*   The navbar now contains a button to upload the whole DB from a .csv file.

### Version 0.1.1 (27.10.2024)
Added a way to select multiple files in the 'Images' module to perform some actions on them, such as deleting or moving into another folder.  
Added batch deletion of images in the 'Images' module.  
Added batch moving of images in the 'Images' module.  
Added a way to search for similar images to the selected one in the 'Images' module. To activate it, you need to place path to the image into the search field or press the 'Find similar' button nearby the image.  
If shift is pressed while selecting images, all images between are ether selected or deselected if there were no images selected beforehand. Otherwise all images from the first to the last selected are selected or deselected.   
Now there are numbers of images in each folder and total number of images including subfoldes displayed in the 'Images' module beside the folder name.  

### Version 0.1.0 (21.10.2024)
Images embeddings caching moved from file-based storage solution to the database. While it is significantly increased the size of the database, the retrieval is much faster now and there might be a way to optimize the compression of the embeddings in the future. It also allow to store embeddings of the images that was removed to still use them for the training process.  
In 'Training' module MAPE calculation is slightly change to better represents small values that might results as negatives before and make not much sense as a percentage.  
In 'Images' module now evaluating model hash and embedding model hash are stored in the database to better track where the values comes from and when they need to be updated.  
Gathering image embeddings process was completely rewritten for better performance and to be able to use the new caching mechanism. Now it takes about 1 minute per 10000 images to gather embeddings for the first time.  
Now there is no need to restart the server to update the image embeddings model. It is done automatically when the training is completed.  
It is now possible to update 'Images' media folder from the UI.  
Sort by rating process in 'Images' module was improved and optimized.  
When training process is shown in the UI, decimation is now performed 'manually' as the method integrated in the 'chart.js' does not seems to work with dynamically changing graphs.  
Image module is now fully functional.  
New 'Images' page is added to the wiki with the description of how to use 'Images' module.  

### Version 0.0.16 (12.10.2024)  
Many improvements in file reading and filtering performance on 'Images' module. These improvements will be spread to other modules later in the development.  
Added caching mechanism for hash gathering in 'Images'.  
Added file list caching mechanism to cache current state of each subfolder with respect to its last modification time.  
Added mechanism to train and use evaluation model for images.  
In 'Train' module added a button to activate the image evaluation model training process.  
Improved search by resolution and proportion performance, but limited maximum number of images for processing by 10000 as the result is not yet cached and there is no indication of the current progress.  
Added callback for embedding gathering to better represent internal processes to the user in 'Images'.  
Now pressing enter in the search bar leads to automatic call to search method in 'Images'.  
Model rating in 'Images' now stored as a float in the database to allow more fine-tuned sorting by this value.  

### Version 0.0.15 (07.10.2024)
Added mechanism to rate images with UI and store it in DB in a more efficient way that it was done with music. The music DB should also be change later to follow the same principals.

### Version 0.0.14 (29.09.2024)
Added current number of images processed at the embedding extracting stage, as sometimes it could take some amount of time at first extraction.  
The GLIP model is no longer loaded into RAM before image search methods are called.  
'Music' wiki page is updated with the description of the current version of music recommendation algorithm.  
Added display of LaTeX formulas in markdown files.  
Added 'Videos' page with simple data gathering and generating/displaying preview of videos.  

### Version 0.0.13 (02.07.2024)
Fixed issue when switching pages in images was resetting the current active folder.  
Added button that sends image file into a trash bin.  
Added pagination at the bottom of the page in images.  
Added search status text in images.  
Added full size image preview with PhotoSwipe.  
Implemented fast cache for image embeddings for faster search.  

### Version 0.0.12 (28.06.2024)
Now when a user rates a song the data of the action is also stored in the DB for further analysis.  
Added display of information about the image, such as file path, file size, resolution and so on is now displayed under each image.    
Added button below the image that opens it inside its folder.   
Added search by file size, images resolution and similarity of the images.   
Implemented switching between subfolders in images.  

### Version 0.0.11 (12.06.2024)
Added 'Images' page with text-based image search using [SigLIP](https://arxiv.org/abs/2303.15343) model.  

### Version 0.0.10 (17.05.2024)
The main Readme.md file has been updated to better represent the current status of the project.  
Wiki pages have also been updated to better reflect the project at the current state.  
Removed all CDN services dependencies.  

### Version 0.0.9 (13.05.2024)
Fixed an issue, when music is not readable after indexing (wrong path was written to DB).  
Did some more investigation on how to encode embeddings to text to then train an LLM on top of it to predict user score for a different types of data. Unfortunately, no experiments show better or even comparable results to a simple feed-forward NN train from embeddings directly. So my conclusion for now, that an idea of a singular rating model produced from a pre-trained LLM might be too hard to implement, so I guess the better approach would be to use separate networks for different types of data and maybe find a better way later. For now I'm going to abandon using LLM in the project at all and use some text-embedding model instead.  
Moved the project to extension-based architecture where each such extension might be developed completely separately from other extensions using general methods from the main code base. That is achieved by storing everything that relates to an extension - js, python and html code - in a separate folder in the main 'pages' folder.  
Temporarily removed LLM instance, chat and news pages as a part of rebuilding the project with new architecture.  
Now when the music library is updating the progress bar actually showing the progress of it.  
On the music library page added a way to change the music library folder directly from the UI.  
'Fine-Tuning' page is replaced with 'Train' pages that are now made as an extension and for now only deal with the music evaluator model.  
Now when music is selected in the radio mode it is rescored with the current evaluator audio model in case it was updated but not used to reindex the library yet.  

### Version 0.0.8 (30.04.2024)
Did some research on embedding integration to Llama2 with QLoRa. While the approach itself is working, unfortunately, it showed quite a low performance compared to even the simplest NN. Research data and scripts are stored in 'research/audio_embeddings'.  
Made simple network that predict ratings from audio embeddings, right now it only works with *.mp3 files and treat each rating as a separate class on the training and evaluation stage. In the future the final score will be a weighted sum of the classes and the training loss function might be changed to support that.  
Added button for starting training of the evaluator network at the fine-tuning page.
Now when the music library is updating the model based evaluation of the music rating is also updated with the current evaluation model.  
When music is selected, model-based rating is used in case there is no user rating for a particular song.  
Fronted now shows model-based ratings in case there are no user ratings in light blue color.  

### Version 0.0.7 (01.04.2024)
For song information also added information about the probability with which the song has been selected. Mostly for debugging purposes.   
Added last_played data for songs to the DB and in the song preview.  
Now the new song selection probability also depends on when the last time it was played.  
Now it's possible to change the song by pressing on its cart in the radio mode.  
Now the current state of the song is constantly saved. This allows you to continue listening after the page has been reloaded. 

### Version 0.0.6 (25.02.2024)
Regex format for predicted GPS coordinates of the news has been fixed to allow negative numbers.  
Some code refactoring related to the music page.  
Some preparations and tests for extension-based architecture.  
Added some primitive control of the news sources with config.yaml.  
Added radio configuration form with an option to switch off AI DJ.  

### Version 0.0.5 (14.01.2024)
Added LLM loading indication.  
Added tables markdown extension for proper wiki rendering.  
TTS initialization changed to an on demand approach for faster start-up time.  
Added wiki pages rendering. The 'wiki' button has been removed, as everything is acceptable through the main page. 
Audio elements removed from base.html and added to music.html.
Added message about TTS initialization when AIDJ was activated.
Added music library navigation system.
Added icon change on play button when song is paused. 
Radio related code has been moved to a separate js file, also some code cleanup. 
Now radio history is not lost when the music page has been reloaded.

### Version 0.0.4 (06.01.2024)
Added installation info to the README.md file.  
Fixed some bugs that cause full_play_count and skip_count not being updated.  
Fixed bug that caused page to scroll up when closing song edit modal window.  
Fixed a dramatic mistake in fine-tuning, where text based data were sampled line by line instead of document based sampling as it was intended.  
Added "Self-aware" mode to the fine-tuning that adds the main code of the project to the LLM training data. 
Better representation of radio mode play history.  
Tabs on the music page are responsive now.  
Now TextPredictor is a singleton class that should eliminate problems with copying LLM in several processes.  
Added "The Gardian" news aggregator.  

### Version 0.0.3 (16.11.2023)
Radio mode converted to a state based approach to see everything that is happening at the backend in real time and stack prepared announcements and songs in a buffer.  
Images of song covers are now sent from backend to frontend as base64 images.  
There are now multiple prompts for AI DJ. It should help to avoid repetitions.  
Added an edit button that allows to change song metadata from the frontend UI. Right now only the lyrics of the song are actually changed.  

### Version 0.0.2 (10.11.2023)
Added skip multiplier to music recommendation algorithm.  
Added display of the music cover to the music page.  
Added internal playback history and switching to the previous track.  
Added play button icon change when paused.  
Now you can start playing random songs after the page music loaded by pressing the play button.  
Now you can change current song play time by clicking on the progress bar.  
Added initial version of AI DJ mode.  
Added 'config.yaml' that contains all currently available settings of the application.  

### Initial version 0.0.1 (01.09.2023)
Initial server and frontend architecture implemented.  
Basic news rating using LLAMA 2 implemented.  
Basic music player with user's ratings implemented.  
Front page of the server displays the README.md file.  
