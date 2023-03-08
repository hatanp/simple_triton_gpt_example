#!/bin/bash
export WORKDIR=/media/volume/cache/tmp/
export HF_DATASETS_CACHE=/media/volume/cache/hf/datasets/
export HF_METRICS_CACHE=/media/volume/cache/hf/metrics/
export HF_MODULES_CACHE=/media/volume/cache/hf/modules/
export HF_TRANSFORMERS_CACHE=/media/volume/cache/hf/transformers/
export TRANSFORMERS_CACHE=/media/volume/cache/hf/transformers/
export TMPDIR=/media/volume/cache/tmp

#sudo docker run --shm-size 1g -it --gpus=1 --rm -p8000:8000 -p8001:8001 -p8002:8002 -v/media/volume/triton/model_repository:/models nvcr.io/nvidia/tritonserver:23.01-py3 bash

sudo docker run -it --rm --net=host -v/media/volume/triton/simple_triton_gpt_example/client:/client nvcr.io/nvidia/tritonserver:23.01-py3-sdk python3 /client/gpt_client.py

