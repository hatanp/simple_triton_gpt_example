#!/bin/bash

export WORKDIR=/media/volume/cache/tmp/
export HF_DATASETS_CACHE=/media/volume/cache/hf/datasets/
export HF_METRICS_CACHE=/media/volume/cache/hf/metrics/
export HF_MODULES_CACHE=/media/volume/cache/hf/modules/
export HF_TRANSFORMERS_CACHE=/media/volume/cache/hf/transformers/
export TRANSFORMERS_CACHE=/media/volume/cache/hf/transformers/
export TMPDIR=/media/volume/cache/tmp

python3 setup.py