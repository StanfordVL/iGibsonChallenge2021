#!/usr/bin/env bash

DOCKER_NAME="my_submission"

while [[ $# -gt 0 ]]
do
key="${1}"

case $key in
      --docker-name)
      shift
      DOCKER_NAME="${1}"
      shift
      ;;
    *) # unknown arg
      echo unkown arg ${1}
      exit
      ;;
esac
done


docker run -it -v $(pwd)/gibson-challenge-data:/gibson-challenge-data \
    --runtime=nvidia \
    ${DOCKER_NAME} \
    /bin/bash -c \
    "CHALLENGE_TRACK=social; export LOG_DIR=test; cd /opt/agents/tf_agents/agents/sac/examples/v1; ./train_single_env.sh"
 
