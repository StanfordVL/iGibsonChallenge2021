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


docker run -v $(pwd)/gibson_challenge_data_2021:/opt/iGibson/gibson2/data \
    --runtime=nvidia \
    ${DOCKER_NAME} \
    /bin/bash -c \
    "export CONFIG_FILE=/opt/iGibson/gibson2/examples/configs/locobot_social_nav.yaml; export TASK=social; export SPLIT=minival; export EPISODE_DIR=/opt/iGibson/gibson2/data/episodes_data/social_nav; bash submission.sh"
    
# make sure CONFIG_FILE, TASK and EPISODE_DIR are consistent
# you can use TASK environment variable to switch agent in agent.py