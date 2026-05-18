# To build the Test Docker Image:
docker-compose -f tests/docker-compose.test.yml build

# To Open an Interactive Shell for Manual Testing:
docker-compose -f tests/docker-compose.test.yml run --rm anagnorisis-test

# To Run a Specific Test Script:
docker-compose -f tests/docker-compose.test.yml run --rm anagnorisis-test python3 -m src.text_embedder
docker-compose -f tests/docker-compose.test.yml run --rm anagnorisis-test python3 -m src.image_embedder
docker-compose -f tests/docker-compose.test.yml run --rm anagnorisis-test python3 -m src.audio_embedder
docker-compose -f tests/docker-compose.test.yml run --rm anagnorisis-test python3 -m src.omni_descriptor
docker-compose -f tests/docker-compose.test.yml run --rm anagnorisis-test python3 -m src.universal_evaluator
docker-compose -f tests/docker-compose.test.yml run --rm anagnorisis-test python3 -m modules.text.engine
docker-compose -f tests/docker-compose.test.yml run --rm anagnorisis-test python3 -m modules.images.engine
docker-compose -f tests/docker-compose.test.yml run --rm anagnorisis-test python3 -m modules.music.engine
docker-compose -f tests/docker-compose.test.yml run --rm anagnorisis-test python3 -m src.share_api