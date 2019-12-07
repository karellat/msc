#!/bin/bash
#PBS -N Pipeline
#PBS -l select=mem=120gb:scratch_local=10gb:ngpus=1:cluster=adan

# Tensorflow module adds cuda drive, cudnn
module add tensorflow-2.0.0-gpu-python3

cd /storage/praha1/home/karellat/master_thesis

python src/pipeline.py
