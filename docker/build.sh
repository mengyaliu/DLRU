#!/usr/bin/env bash

#
# Execute command within a docker container
#
# Usage: build.sh <CONTAINER_TYPE> [Version]
#
# CONTAINER_TYPE: Type of the docker container used the run the build: e.g.,
#                 (cpu | gpu)
# Version: define the docker version
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Get the command line arguments.
CONTAINER_TYPE=$( echo "$1" | tr '[:upper:]' '[:lower:]' )
shift 1
VERSION=$1
if [ "$VERSION" = "" ]; then VERSION=latest; fi

# Dockerfile to be used in docker build
DOCKERFILE_PATH="${SCRIPT_DIR}/Dockerfile.${CONTAINER_TYPE}"
DOCKER_CONTEXT_PATH="${SCRIPT_DIR}"

if [[ ! -f "${DOCKERFILE_PATH}" ]]; then
    echo "Invalid Dockerfile path: \"${DOCKERFILE_PATH}\""
    exit 1
fi

# Validate command line arguments.
if [ ! -e "${SCRIPT_DIR}/Dockerfile.${CONTAINER_TYPE}" ]; then
    supported_container_types=$( ls -1 ${SCRIPT_DIR}/Dockerfile.* | \
        sed -n 's/.*Dockerfile\.\([^\/]*\)/\1/p' | tr '\n' ' ' )
      echo "Usage: $(basename $0) CONTAINER_TYPE Version"
      echo "       CONTAINER_TYPE can be one of [${supported_container_types}]"
      exit 1
fi

# Use nvidia-docker if the container is GPU.
if [[ "${CONTAINER_TYPE}" == *"gpu"* ]]; then
    DOCKER_BINARY="nvidia-docker"
else
    DOCKER_BINARY="docker"
fi

# Helper function to traverse directories up until given file is found.
function upsearch () {
    test / == "$PWD" && return || \
        test -e "$1" && echo "$PWD" && return || \
        cd .. && upsearch "$1"
}

# Set up WORKSPACE and BUILD_TAG. Jenkins will set them for you or we pick
# reasonable defaults if you run it outside of Jenkins.
WORKSPACE="${WORKSPACE:-${SCRIPT_DIR}/../}"
BUILD_TAG="${BUILD_TAG:-dlru}"

# Determine the docker image name
DOCKER_IMG_NAME="${BUILD_TAG}/${CONTAINER_TYPE}"

# Under Jenkins matrix build, the build tag may contain characters such as
# commas (,) and equal signs (=), which are not valid inside docker image names.
DOCKER_IMG_NAME=$(echo "${DOCKER_IMG_NAME}" | sed -e 's/=/_/g' -e 's/,/-/g')

# Convert to all lower-case, as per requirement of Docker image names
DOCKER_IMG_NAME=$(echo "${DOCKER_IMG_NAME}" | tr '[:upper:]' '[:lower:]')

# Print arguments.
echo "WORKSPACE: ${WORKSPACE}"
echo "CI_DOCKER_EXTRA_PARAMS: ${CI_DOCKER_EXTRA_PARAMS[@]}"
echo "CONTAINER_TYPE: ${CONTAINER_TYPE}"
echo "BUILD_TAG: ${BUILD_TAG}"
echo "DOCKER CONTAINER NAME: ${DOCKER_IMG_NAME}"
echo ""

# Build the docker container.
echo "Building container (${DOCKER_IMG_NAME})..."
docker build -t ${DOCKER_IMG_NAME}:${VERSION} \
    -f "${DOCKERFILE_PATH}" "${DOCKER_CONTEXT_PATH}"

# Check docker build status
if [[ $? != "0" ]]; then
    echo "ERROR: docker build failed."
    exit 1
fi

