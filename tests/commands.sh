# To Open an Interactive Shell for Manual Testing:
docker-compose -f tests/docker-compose.test.yml run --rm anagnorisis-test

# To Run a Specific Test Script:
docker-compose -f tests/docker-compose.test.yml run --rm anagnorisis-test python3 -m src.text_embedder
docker-compose -f tests/docker-compose.test.yml run --rm anagnorisis-test python3 -m pages.text.engine