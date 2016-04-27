#!/bin/bash
#PBS -q isi
#PBS -l walltime=336:00:00
#PBS -l gpus=2

set -e

#This script was written by barret zoph for questions email barretzoph@gmail.com
#It will return 1 if not successful, 0 if successful

#### Things that must be specified by user ####
SOURCE_TRAIN_FILE=$1 #Hard path and name of the source file being rescored
TARGET_TRAIN_FILE=$2 #Hard path and name of the target file being rescored
DIRECTORY=$3 #Location for where the model will be trained
SOURCE_DEV_FILE=$4 #Source dev file
TARGET_DEV_FILE=$5 #Target dev file
MODEL_OPTS=$6 #Model options
GPU_OPTS=$7 #What layers should be on what gpus
SHARED_OPTS=$8 #More model options
RNN_LOCATION=$9 #Path to executable

#assumptions this file makes

echo "SOURCE_TRAIN_FILE = $SOURCE_TRAIN_FILE"
echo "TARGET_TRAIN_FILE = $TARGET_TRAIN_FILE"
echo "DIRECTORY = $DIRECTORY"
echo "SOURCE_DEV_FILE = $SOURCE_DEV_FILE"
echo "TARGET_DEV_FILE = $TARGET_DEV_FILE"
echo "MODEL_OPTS = $MODEL_OPTS"
echo "GPU_OPTS = $GPU_OPTS"
echo "SHARED_OPTS = $SHARED_OPTS"
echo "RNN_LOCAION = $RNN_LOCATION"

#################### NOTHING BELOW NEEDS TO BE CHANGED ####################

#### Run some error checks ####



#### Sets up environment to run code ####
source /usr/usc/cuda/7.0/setup.sh
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/nlg-05/zoph/cudnn_v4/lib64/
export LD_LIBRARY_PATH

cd $DIRECTORY
cmd="$RNN_LOCATION -t $SOURCE_TRAIN_FILE $TARGET_TRAIN_FILE model.nn -a $SOURCE_DEV_FILE $TARGET_DEV_FILE $MODEL_OPTS $SHARED_OPTS $GPU_OPTS --train-ensemble ../count6.nn"
echo $cmd;
$cmd;

exit 0

