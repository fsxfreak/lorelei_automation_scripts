#!/bin/bash
#PBS -q isi
#PBS -l walltime=336:00:00
#PBS -l gpus=2

cd $1 
source /usr/usc/cuda/7.0/setup.sh
bash align unk_replace.conf
