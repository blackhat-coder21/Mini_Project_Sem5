#!/bin/bash

#PBS -u anupam
#PBS -N jupyterJob
#PBS -q gpu
#PBS -l select=1:ncpus=40:ngpus=2
#PBS -o out.log
#PBS -j oe
#PBS -V

module load compilers/intel/parallel_studio_xe_2018_update3_cluster_edition
module load codes/cuda/cuda-11.8

export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0,1
cd $PBS_O_WORKDIR
nohup jupyter notebook --no-browser --ip=$(hostname) --port=8888 --NotebookApp.token='' --NotebookApp.password='' > jupyter.log 2>&1 &

sleep 1000000000