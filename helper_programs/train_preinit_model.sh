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
PRETRAIN_PATH=$6 #location of the pretrain.pl script

#assumptions this file makes
#1. This model is an attention model using feed-input

#################### NOTHING BELOW NEEDS TO BE CHANGED ####################

#### Run some error checks ####



#### Sets up environment to run code ####
source /usr/usc/cuda/7.5/setup.sh
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/usc/cuDNN/7.5-v5.1/lib64/
export LD_LIBRARY_PATH

cd $DIRECTORY
perl $PRETRAIN_PATH parent.nn $SOURCE_TRAIN_FILE $TARGET_TRAIN_FILE $SOURCE_DEV_FILE $TARGET_DEV_FILE best.nn

exit 0
 
