## Automated Tests

Automated tests live in `tests/` and are split into tiers based on their hardware requirements. See `tests/commands.sh` for exact Docker run commands.

### Tier 1 — Pure-logic tests (no GPU, no model downloads)

Run the full suite in one command:
```
docker-compose -f tests/docker-compose.test.yml run --rm anagnorisis-test pytest tests/ -v
```

| Test file | What it covers |
|-----------|----------------|
| `test_config_loader.py` | `${VAR:-default}` substitution, plain `$VAR`, missing env vars, nested YAML, invalid YAML |
| `test_caching.py` | `RAMCache` TTL & thread-safety; `DiskCache` round-trip, TTL, corrupted-shard recovery, atomic writes, warming callback, write-back; `TwoLevelCache` RAM-first / disk-fallback |
| `test_embedding_proxy.py` | `quantize_embedding()` — zero embedding, output length, alphabet, histogram-equalisation, similar/orthogonal embeddings, CLAP (512-dim) and SigLIP (768-dim) |
| `test_file_manager.py` | `resolve_subpath()` — valid paths, None/empty, `../`, multi-hop, URL-decoded & double-encoded traversal, absolute escape, symlinks; `get_folder_structure()` — missing dir, extension counting, subfolder totals |
| `test_common_filters.py` | `_normalize_text()` — accent stripping, separator normalisation, case folding; `filter_by_text(mode='file-name')` — exact-match boost, fuzzy match, unicode, empty query, unknown mode |
| `test_task_manager.py` | Task submission, FIFO order, exception handling (worker survives), cancel running/queued tasks, pause/resume, history, `get_state()` structure |
| `test_db_models.py` | `export_db_to_csv()` — header format, row count, excluded columns, datetime; `import_db_from_csv()` — new rows, update-by-hash, unknown columns skipped, round-trip, empty CSV |

### Tier 3 — Security tests (no GPU required)

```
docker-compose -f tests/docker-compose.test.yml run --rm anagnorisis-test pytest tests/test_security_path_traversal.py -v
```

`test_security_path_traversal.py` covers:
- `_looks_like_path` / `_has_parent_segment` helper logic
- Flask `before_request` middleware: query params, JSON body (nested & list), form data
- All standard traversal variants: `../`, `%2e%2e/`, `%252e%252e/`, backslash, mixed-case `/ETC/`
- `/etc/` and `/proc/` blocked; safe paths (normal filenames, image paths) pass through

### Tier 2 — ML model tests (requires GPU + model downloads)

These run the `__main__` blocks of each subprocess worker to verify model loading and inference produce valid output.

```
python3 -m src.text_embedder
python3 -m src.image_embedder
python3 -m src.audio_embedder
python3 -m src.omni_descriptor
python3 -m src.universal_evaluator
python3 -m modules.text.engine
python3 -m modules.images.engine
python3 -m modules.music.engine
python3 -m modules.videos.engine   # no embedding model; tests metadata extraction
python3 -m src.recommendation_engine
python3 -m src.share_api
```

### TODO — Structural improvements (future work)

- **Migrate `__main__` scripts to pytest** so model tests produce structured pass/fail output and can be filtered with `-k`.
- **Shared test fixtures** — create `tests/fixtures/` with one real JPEG, WAV, and TXT file reused across all engine tests instead of generating synthetic data per test.
- **Two-tier CI** — run Tier 1 & 3 tests in GitHub Actions on every push (no GPU needed); keep Tier 2 as manual Docker-only tests.
- **`modules/videos/engine.py` `__main__` test** — videos module is the only engine not yet covered by a Tier 2 run script; add metadata-extraction test similar to images/music.
- **`src/metadata_search.py` integration test** — test the full `generate_full_description()` pipeline (metadata → proxy → Jina → description) on a known file, verifying caching on second call.

---

## Manual Clean-state Test

* Clone the current repository state to a test folder.
```
mkdir -p ../Anagnorisis-test
git ls-files -z | rsync -av --files-from=- --from0 ./ ../Anagnorisis-test
```

* Go to the test folder.
```
cd ../Anagnorisis-test
```

* Create a `docker-compose.override.yaml` in the test folder pointing to your test data:
```
cp docker-compose.override.example.yaml docker-compose.override.yaml
```
Then edit it to use a dedicated test config folder and test media folders. Use a **different port** (e.g. `5005`) and a **different container name** so it does not collide with the production instance. For example:
```yaml
services:
  anagnorisis:
    container_name: anagnorisis-app-test
    ports:
      - "127.0.0.1:5005:5001"
    volumes:
      - /path/to/your/test-config:/mnt/project_config
      - /path/to/your/test-images:/mnt/media/images/TestImages
      - /path/to/your/test-music:/mnt/media/music/TestMusic
      - /path/to/your/test-text:/mnt/media/text/TestText
      - /path/to/your/test-videos:/mnt/media/videos/TestVideos
```
Make sure the config folder (`/path/to/your/test-config`) exists on the host before starting — Docker may fail to create it due to permission constraints.

* Run the Docker container.
```
docker compose up -d --build
```

* After the container has been build successfully, open specified `http://localhost:{EXTERNAL_PORT}` in your web browser to see that initialization process is properly displayed and all the on going initialization steps are shown.

* Wait until all the models are downloaded and the application is fully started. Watch the progress in the `logs/{CONTAINER_NAME}_log.txt` file. Or even better break the downloading process by stopping the container and make sure that all the corrupted models are correctly identified and re-downloaded upon the next start.

* Check that all the modules are opens and show their files correctly.

* After opening all the modules, check the logs to make sure there were no silent errors.

* Perform "file-name-based", "semantic-based" and "metadata-based" searches in all module. Make sure no errors happened in the process.

* Check that all recommendation models could be trained without any errors on the `Train` page for each module.

## Caution
In case there is any changes in the codebase while testing, **do not forget** to update the code from the main project folder to the test folder again by running:
```
git ls-files -z | rsync -av --files-from=- --from0 ./ ../Anagnorisis-test
```

And restart the Docker:
```
docker compose restart
```