#!/bin/bash
#PBS -q isi
#PBS -l walltime=336:00:00
#PBS -l gpus=2

tmpdir=${TMPDIR:-/tmp}
MTMP=$(mktemp -d --tmpdir=$tmpdir XXXXXX)
function cleanup() {
    rm -rf $MTMP;
}
trap cleanup EXIT


#### Sets up environment to run code ####
source /usr/usc/cuda/7.0/setup.sh
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/nlg-05/zoph/cudnn_v4/lib64/
export LD_LIBRARY_PATH

FINAL_ARGS=$1
echo "FINAL_ARGS = $FINAL_ARGS"
cd $MTMP/
$FINAL_ARGS
exit 0
