Environment assumptions:

* python3 3.5 or greater (add source /usr/usc/python/default/setup.sh to .bashrc)
* add export LANG=en_US.UTF-8 to .bashrc; otherwise LANG=C on new jobs
* access to qsubrun in path (included here)

rough notes:

* train_and_decode = make standalone, parent, child models given parallel data
* rescore_all = apply an nmt model to sbmt output, represented as a directory
* data for train_and_decode: ~jonmay/projects/cnmt/bz/dr_17_06/[il3,rus]/data

here's a run of train_and_decode that you should be able to run yourself:

scripts/train_and_decode.py --name il3_auto_eighth --lang il3 --standalone dr_17_06/il3/standalone_auto_quick --parent dr_17_06/il3/parent_eighth --child dr_17_06/il3/child_eighth --epochs 20 -P 1 -N 1 --data dr_17_06/il3/data --parent_data parent/fra/data --parent_source train.src.eighth.tc --parent_target train.trg.eighth.tc --parent_dev_source dev.src.tc --parent_dev_target dev.trg.tc --decodes test syscomb dev eval


use -h first to understand what all those options mean; you can clearly change some of them yourself

you shouldn't have to prepare reranking input directories yourself, but here's how you do it. The source material for them comes to me from michael pust; he follows a particular directory convention as well, which this script relies upon: ~jonmay/projects/cnmt/bz/scripts/setup_rerank_input.sh <inputdir> <outputdir>


e.g.  scripts/setup_rerank_input.sh /home/nlg-05/pust/elisa-y2/dryrun-2017-06-05/il3-eng/tl-convert4/penn/usoov rerank/dr_17_06/penn-oov2/il3/input

here's an example of rescore_all: for i in quarter eighth; do scripts/rescore_all.py -i rerank/dr_17_06/penn/il3/input/ -m dr_17_06/il3/child_"$i" -n 1 -L il3 -l rr_"$i" -r dr_17_06/il3/rerank_penn_"$i" -S m_1 -w 10 -e eval; done


Zoph RNN code under repo at git@github.com:isi-nlp/Zoph_RNN.git