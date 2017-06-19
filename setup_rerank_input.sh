#!/usr/bin/env bash

# given pust directory, create directory of simlinks and processed data that is usable by my rescoring script
# weights
# raw data
# nbest for each set
SCRIPTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

PUSTDIR=$1;
OUTDIR=$2;
SETS=(test dev syscomb eval)
mkdir -p $OUTDIR;
PUSTDIR=$(readlink -f $PUSTDIR);
tunedir=$PUSTDIR/tune-dev
if [[ ! -e $tunedir ]]; then
    echo "Couldn't find $tunedir";
    exit 1
fi
ln -s $tunedir/weights.final $OUTDIR/weights.final;
echo $PUSTDIR;
findup=$($SCRIPTDIR/findup.sh $PUSTDIR -type d -name data)
datadir=$(echo $findup | cut -d' ' -f1)
echo $datadir
for set in ${SETS[@]}; do
    nbest=$PUSTDIR/decode-dev-$set.final/nbest.sort
    ref=$datadir/$set.target.tok.tc
    src=$datadir/$set.source.tok.tc
    if [[ ! -e $nbest ]]; then
	echo "Skipping $set; no nbest ($nbest)"
	continue
    fi
    if  [[ ! -e $src ]]; then
	echo "Skipping $set; no src ($src)"
	continue
    fi
    ln -s $nbest $OUTDIR/$set.nbest
    $SCRIPTDIR/nbest2rerankdata.py -i $OUTDIR/$set.nbest -s $src -o $OUTDIR/$set.src.hyp
    if [[ -e $ref ]]; then
	ln -s $ref $OUTDIR/$set.trg.ref;
    else
	echo "No reference for $ref; hope that's ok (e.g. it's eval)"
    fi
done



