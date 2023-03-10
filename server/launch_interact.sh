#!/bin/bash
#This script is only used for running shell in docker to not lose status in development

#Setting various env variables that might or might not be required
sudo docker run --env-file env_var_list.txt --shm-size 1g -it --gpus=1 --rm -p8000:8000 -p8001:8001 -p8002:8002 -v/media/volume/triton/simple_triton_gpt_example/server/model_repository:/models -v/media/volume/cache:/tmp/host nvcr.io/nvidia/tritonserver:23.01-py3 bash

#Run following as you please
#pip install transformers torch diffusers accelerate && tritonserver --model-repository=/models