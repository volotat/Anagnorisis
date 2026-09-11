## Automated Tests

Automated tests live in `tests/` and are split into tiers based on their hardware requirements. See `tests/commands.sh` for exact Docker run commands.

### Two suites, split along the engine boundary

`pytest.ini` points at both, so a bare `pytest` runs everything. Running one at a
time tells you which side of the boundary a failure is on.

**`anagnorisis_core/tests/` — the engine.** These must pass with no Flask, no
database and no browser present, because that is exactly the situation anyone
who installs the package on its own is in — the data server included. Verified
by copying only
`anagnorisis_core/` into an empty tree and running them there.

| Test file | Covers |
|---|---|
| `test_caching.py` | `RAMCache` TTL & thread-safety; `DiskCache` round-trip, TTL, corrupt-shard recovery (live cache, fresh reader, and reusability afterwards), atomic writes, warming callback, write-back; `TwoLevelCache` RAM-first / disk-fallback |
| `test_cache_multiprocess.py` | Two real processes writing one cache directory: neither loses its entries. Guards the read-modify-write race that file locking now prevents |
| `test_core_boundary.py` | No module under `anagnorisis_core` imports `src`, Flask or SQLAlchemy — by AST, so a mention in a docstring does not count. Also that the media-type taxonomy ships with the package |
| `test_model_hash.py` | The model fingerprint is stable across processes, changes when weights/task/dimension change, and is computable **without loading the model** — the property the search path depends on |
| `test_embedder_no_network.py` | The embedding model never fetches a URL it finds inside text. Includes a test that the *unpatched* code does reach the network, so the suite cannot pass vacuously |
| `test_description.py` | The one canonical description: section order, the injected describer, the remote-file branch, the data server's two omissions, and the sidecar reader's caps |
| `test_api_search_and_ratings.py` | Name search without an index; unindexed files dropped rather than ranked last; results are VFS URLs; hidden directories skipped; remote paths refused rather than downloaded; ratings survive a rename; your rating outranks the model's |
| `test_sinks_and_describe.py` | `.meta` is never overwritten, deletion regenerates it, no path is published, no `.partial` files remain, and a file still being copied is not described |
| `test_training_pairs.py` | Reading memory files: the rating comes off line 1 and never reaches the embedder; the gather loop reads every usable file |
| `test_metadata_proxy.py` | `quantize_embedding()` — zero embedding, output length, alphabet, histogram equalisation, similar/orthogonal embeddings |
| `test_file_paths.py` | `resolve_subpath()` — `../`, multi-hop, URL-encoded and double-encoded traversal, absolute escape, symlinks; `get_folder_structure()` |
| `test_common_filters.py` | `_normalize_text()` — accents, separators, case folding; `filter_by_text(mode='file-name')` |

**`tests/` — the application, and the seam.**

| Test file | Covers |
|---|---|
| `test_config_loader.py` | `${VAR:-default}` substitution, plain `$VAR`, missing env vars, nested YAML, invalid YAML |
| `test_task_manager.py` | Task submission, FIFO order, worker survives an exception, cancel, pause/resume, history, `get_state()` |
| `test_db_models.py` | `export_db_to_csv()` / `import_db_from_csv()` round-trips |
| `test_description_consumers.py` | The seam: the text search embeds, the text the UI shows, and a memory file's body are one string |
| `test_core_reexports.py` | The application's re-exports of moved engine code still resolve, and its soft hash is identical to the engine's |
| `test_routes_smoke.py` | Every page answers without a 500. **Opt-in** (`ANAGNORISIS_ROUTE_TESTS=1`): building the real app leaves background threads running, so pytest would not exit |

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
python3 -m anagnorisis_core.models.descriptor
python3 -m anagnorisis_core.models.universal_evaluator
python3 -m anagnorisis_core.search.recommendation_engine
```

There is no `src.share_api` any more. Serving is `rclone` and describing a shared
folder is the command line, so the data server has no subprocess worker of its
own to self-test — see `data_server/README.md`.

The three per-modality embedders and the four per-module engines no longer exist —
one model and one engine replaced them. They have no `__main__` self-tests yet;
what is worth covering instead is:

- `anagnorisis_core.models.embedder` — load the model, embed one file of each media type, check
  the dimension and that the CPU query tower agrees with the GPU worker (they
  must produce interchangeable vectors, or search silently stops matching).
- `anagnorisis_core.search.content_search` — embed and compare a known file; confirm the cache key
  written matches the one the embedding proxy rebuilds.

### TODO — Structural improvements (future work)

- **Migrate `__main__` scripts to pytest** so model tests produce structured pass/fail output and can be filtered with `-k`.
- **Shared test fixtures** — create `tests/fixtures/` with one real JPEG, WAV, and TXT file reused across all engine tests instead of generating synthetic data per test.
- **Two-tier CI** — run Tier 1 & 3 tests in GitHub Actions on every push (no GPU needed); keep Tier 2 as manual Docker-only tests.
- **`anagnorisis_core/models/embedder.py` self-test** — the checks listed under Tier 2 above; the CPU/GPU agreement check in particular is load-bearing and currently only verified by hand.
- **`anagnorisis_core/search/metadata_search.py` integration test** — the full `generate_full_description()` pipeline (extractor → proxy → embedder → description) on a known file, verifying caching on the second call.
- **A GPU-isolation regression test** — assert that no search path invokes the GPU worker. This is easy to check by stubbing `OmniEmbedder._execute` to raise, and easy to break accidentally.
- **`anagnorisis_core/media/media_types.py` coverage** — the registry rejects duplicate extensions across types and derives each module's `media_formats`; both are startup-critical and untested.

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