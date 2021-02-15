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


docker run -v $(pwd)/gibson-challenge-data:/gibson-challenge-data \
    --runtime=nvidia \
    ${DOCKER_NAME} \
    /bin/bash -c \
    "export CONFIG_FILE=/gibson-challenge-data/locobot_p2p_nav_house.yaml; export SIM2REAL_TRACK=static; cp /gibson-challenge-data/global_config.yaml /opt/GibsonEnvV2/gibson2/global_config.yaml; bash submission.sh"
