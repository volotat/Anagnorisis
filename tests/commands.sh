# To build the Test Docker Image:
docker compose -f tests/docker-compose.test.yml build

# To Open an Interactive Shell for Manual Testing:
docker compose -f tests/docker-compose.test.yml run --rm anagnorisis-test

# ─────────────────────────────────────────────────────────────────────────────
# The suite is split in two, and which half fails tells you where the problem is:
#
#   anagnorisis_core/tests   the engine. Must pass with no Flask, no database and
#                            no browser — the same boundary the annotator relies on.
#   tests                    the Flask application, plus the few tests that sit on
#                            the seam between it and the engine.
#
# Everything (pytest.ini points at both):
docker compose -f tests/docker-compose.test.yml run --rm anagnorisis-test pytest -q
#
# One side at a time:
docker compose -f tests/docker-compose.test.yml run --rm anagnorisis-test pytest anagnorisis_core/tests -q
docker compose -f tests/docker-compose.test.yml run --rm anagnorisis-test pytest tests -q

# Individual modules, engine side:
docker compose -f tests/docker-compose.test.yml run --rm anagnorisis-test pytest anagnorisis_core/tests/test_caching.py -v
docker compose -f tests/docker-compose.test.yml run --rm anagnorisis-test pytest anagnorisis_core/tests/test_metadata_proxy.py -v
docker compose -f tests/docker-compose.test.yml run --rm anagnorisis-test pytest anagnorisis_core/tests/test_core_boundary.py -v

# Individual modules, application side:
docker compose -f tests/docker-compose.test.yml run --rm anagnorisis-test pytest tests/test_config_loader.py -v
docker compose -f tests/docker-compose.test.yml run --rm anagnorisis-test pytest anagnorisis_core/tests/test_model_hash.py -v
docker compose -f tests/docker-compose.test.yml run --rm anagnorisis-test pytest anagnorisis_core/tests/test_file_paths.py -v
docker compose -f tests/docker-compose.test.yml run --rm anagnorisis-test pytest anagnorisis_core/tests/test_common_filters.py -v
docker compose -f tests/docker-compose.test.yml run --rm anagnorisis-test pytest tests/test_task_manager.py -v
docker compose -f tests/docker-compose.test.yml run --rm anagnorisis-test pytest tests/test_db_models.py -v

# Route smoke tests are opt-in: building the real app leaves background threads
# running, so pytest would not exit. Run them in their own process:
docker compose -f tests/docker-compose.test.yml run --rm -e ANAGNORISIS_ROUTE_TESTS=1 anagnorisis-test pytest tests/test_routes_smoke.py -v

# Tier 3 — Security tests (no GPU required):
docker compose -f tests/docker-compose.test.yml run --rm anagnorisis-test pytest tests/test_security_path_traversal.py -v

# ─────────────────────────────────────────────────────────────────────────────
# Tier 2 — ML model tests (requires GPU + model downloads)
# One embedder now covers every media type, so there is one script to run
# instead of one per modality:
docker compose -f tests/docker-compose.test.yml run --rm anagnorisis-test python3 -m anagnorisis_core.models.embedder
docker compose -f tests/docker-compose.test.yml run --rm anagnorisis-test python3 -m anagnorisis_core.models.descriptor
docker compose -f tests/docker-compose.test.yml run --rm anagnorisis-test python3 -m src.universal_evaluator
docker compose -f tests/docker-compose.test.yml run --rm anagnorisis-test python3 -m src.share_api
docker compose -f tests/docker-compose.test.yml run --rm anagnorisis-test python3 -m src.recommendation_engine