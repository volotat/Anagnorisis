# To build the Test Docker Image:
docker-compose -f tests/docker-compose.test.yml build

# To Open an Interactive Shell for Manual Testing:
docker-compose -f tests/docker-compose.test.yml run --rm anagnorisis-test

# ─────────────────────────────────────────────────────────────────────────────
# Tier 1 — Pure-logic tests (no GPU, no models required)
# Run all at once with pytest (recommended):
docker-compose -f tests/docker-compose.test.yml run --rm anagnorisis-test pytest tests/ -v --ignore=tests/docker-compose.test.yml --ignore=tests/Dockerfile.test

# Or run individual test modules:
docker-compose -f tests/docker-compose.test.yml run --rm anagnorisis-test pytest tests/test_config_loader.py -v
docker-compose -f tests/docker-compose.test.yml run --rm anagnorisis-test pytest tests/test_caching.py -v
docker-compose -f tests/docker-compose.test.yml run --rm anagnorisis-test pytest tests/test_embedding_proxy.py -v
docker-compose -f tests/docker-compose.test.yml run --rm anagnorisis-test pytest tests/test_file_manager.py -v
docker-compose -f tests/docker-compose.test.yml run --rm anagnorisis-test pytest tests/test_common_filters.py -v
docker-compose -f tests/docker-compose.test.yml run --rm anagnorisis-test pytest tests/test_task_manager.py -v
docker-compose -f tests/docker-compose.test.yml run --rm anagnorisis-test pytest tests/test_db_models.py -v

# Tier 3 — Security tests (no GPU required):
docker-compose -f tests/docker-compose.test.yml run --rm anagnorisis-test pytest tests/test_security_path_traversal.py -v

# ─────────────────────────────────────────────────────────────────────────────
# Tier 2 — ML model tests (requires GPU + model downloads)
# Run a specific model/engine script:
docker-compose -f tests/docker-compose.test.yml run --rm anagnorisis-test python3 -m src.text_embedder
docker-compose -f tests/docker-compose.test.yml run --rm anagnorisis-test python3 -m src.image_embedder
docker-compose -f tests/docker-compose.test.yml run --rm anagnorisis-test python3 -m src.audio_embedder
docker-compose -f tests/docker-compose.test.yml run --rm anagnorisis-test python3 -m src.omni_descriptor
docker-compose -f tests/docker-compose.test.yml run --rm anagnorisis-test python3 -m src.universal_evaluator
docker-compose -f tests/docker-compose.test.yml run --rm anagnorisis-test python3 -m modules.text.engine
docker-compose -f tests/docker-compose.test.yml run --rm anagnorisis-test python3 -m modules.images.engine
docker-compose -f tests/docker-compose.test.yml run --rm anagnorisis-test python3 -m modules.music.engine
docker-compose -f tests/docker-compose.test.yml run --rm anagnorisis-test python3 -m modules.videos.engine
docker-compose -f tests/docker-compose.test.yml run --rm anagnorisis-test python3 -m src.share_api
docker-compose -f tests/docker-compose.test.yml run --rm anagnorisis-test python3 -m src.recommendation_engine