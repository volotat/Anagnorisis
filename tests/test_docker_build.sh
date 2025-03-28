#!/bin/bash
set -e

IMAGE_NAME="anagnorisis-test"
CONTAINER_NAME="anagnorisis-test-container"
DOCKERFILE_PATH="tests/dockerfile.test" # Define Dockerfile path


echo "Building Docker image..."
# Use process substitution to pipe tar archive of git-tracked files to docker build
docker build -f tests/dockerfile.test -t ${IMAGE_NAME} - < <(
  cd .. && (
    git ls-files -c -o --exclude-standard ;
    find models/siglip-base-patch16-224 -type f ;
    find models/clap-htsat-fused -type f
  ) | tar -czf - -T -
)

echo "Running Docker container..."
docker run --rm --name ${CONTAINER_NAME} -d -p 5001:5001 ${IMAGE_NAME}  # Run in detached mode

echo "Waiting for container to start..."
sleep 30 # Give the container some time to start

echo "Checking if application is running..."
# You can customize this check - for now, let's just check if the container is running
container_status=$(docker inspect -f '{{.State.Status}}' ${CONTAINER_NAME})

if [ "${container_status}" == "running" ]; then
  echo "Container is running (Status: ${container_status})"
  echo "Application seems to have started successfully (basic check)."
  test_result="pass"
else
  echo "Container status: ${container_status}"
  echo "Application may have failed to start."
  test_result="fail"
fi

# Now check that connecting to http://localhost:5001/ returns HTTP 200
http_code=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:5001/)
echo "HTTP response code: ${http_code}"
if [ "$http_code" == "200" ]; then
  echo "Endpoint http://localhost:5001/ is healthy (HTTP 200)."
  test_result="pass"
else
  echo "Endpoint http://localhost:5001/ returned an error (HTTP ${http_code})."
  test_result="fail"
fi

echo "Stopping and removing container..."
docker stop ${CONTAINER_NAME}
#docker rm ${CONTAINER_NAME}

if [ "${test_result}" == "pass" ]; then
  echo "Docker-based test PASSED!"
  exit 0  # Indicate success
else
  echo "Docker-based test FAILED!"
  exit 1  # Indicate failure
fi
 