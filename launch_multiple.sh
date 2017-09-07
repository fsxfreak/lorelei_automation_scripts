#!/bin/bash

# a little helper to launch multiple decodes when testing multiple train runs

for i in `seq 0 7`;
do
     #./train_and_decode.py \
     #    --name de-sa-$i --lang il3 \
     #    --epochs 20 -P 1 -N 1 \
     #   -p null -c null --parent_data null \
     #   --standalone ../../output/de-sa-$i \
     #   --no-do_parent --no-do_child  --no-do_standalone_package --no-do_standalone_decode \
     #   --data /home/nlg-05/ljcheung/data/ep-de-en \
     #   --decodes test
    ./train_and_decode.py \
        --standalone ../../output/de-sa-$i \
        --name de-sa-$i --lang il3 \
        --epochs 20 -P 1 -N 1 \
        -p null -c null --parent_data null \
        --no-do_parent --no-do_child  --no-do_standalone_train --no-do_standalone_package \
        --data /home/nlg-05/ljcheung/data/ep-de-en \
        --decodes test
done
