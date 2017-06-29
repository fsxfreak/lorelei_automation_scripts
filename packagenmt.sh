#!/usr/bin/env bash
#PBS -q isi
#PBS -l walltime=1:00:00


# packages up decoded nmt run by decode_models.sh
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )""/"
curr=/home/nlg-02/pust/pipeline-2.20;
decfile=$1;
origcorpus=$2;
tstmaster=$3;
ofile=$4;

$curr/scripts/detok $decfile $origcorpus | $curr/bin/lc | $DIR/xmlify-nbest -t SOURCE <(zcat $tstmaster) | gzip > $ofile;
