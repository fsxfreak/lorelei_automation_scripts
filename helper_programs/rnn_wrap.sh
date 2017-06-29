#!/bin/bash
#PBS -q isi
#PBS -l walltime=336:00:00
#PBS -l gpus=2

# just set up env and add a temp dir spec

set -e

tmpdir=${TMPDIR:-/tmp}
MTMP=$(mktemp -d --tmpdir=$tmpdir XXXXXX)
function cleanup() {
    rm -rf $MTMP;
}
trap cleanup EXIT

# 

#### Sets up environment to run code ####
source /usr/usc/cuda/7.5/setup.sh
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/usc/cuDNN/7.5-v5.1/lib64/
export LD_LIBRARY_PATH

"$@" --tmp-dir-location $MTMP
