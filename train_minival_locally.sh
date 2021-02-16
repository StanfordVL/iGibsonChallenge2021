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


docker run -it -v $(pwd)/gibson_challenge_data_2021:/opt/iGibson/gibson2/data \
    --runtime=nvidia \
    ${DOCKER_NAME} \
    /bin/bash -c \
    "export CONFIG_FILE=/opt/iGibson/gibson2/examples/configs/locobot_social_nav.yaml; export LOG_DIR=test; cd /opt/agents/tf_agents/agents/sac/examples/v1; ./train_minival.sh"
 
