#!/bin/bash
srun --partition=MoE --mpi=pmi2 --nodes=1 --cpus-per-task=16 --gres=gpu:1 jupyter lab --notebook-dir=/mnt/petrelfs/shengli/transformer --ip=0.0.0.0 --port=10049