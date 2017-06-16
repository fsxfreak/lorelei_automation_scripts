#!/usr/bin/env python3
# code by Jon May [jonmay@isi.edu]
# prep data, train models, run decodes

import argparse
import sys
import codecs
if sys.version_info[0] == 2:
  from itertools import izip
else:
  izip = zip
from collections import defaultdict as dd
import re
import os
import os.path
import gzip
import tempfile
import shutil
import atexit
import shlex
from subprocess import run, PIPE
from jmutil import mkdir_p
scriptdir = os.path.dirname(os.path.abspath(__file__))

reader = codecs.getreader('utf8')
writer = codecs.getwriter('utf8')


def _standalone(args):
  # train
  cmd = "{train} --name {name} --mode standalone --trained_model {stand} --model_nums {nums} -ts {data}/{tr_src} -tt {data}/{tr_trg} -ds {data}/{dv_src} -dt {data}/{dv_trg} -e {epochs}".format(train=args.traincmd, name=args.name, stand=args.standalone, nums=' '.join(args.model_nums), data=args.data, tr_src=args.train_source, tr_trg=args.train_target, dv_src=args.dev_source, dv_trg=args.dev_target, epochs=args.epochs)
  sys.stderr.write(cmd+"\n")
  trainid = ""
  if args.do_standalone_train:
    trainid=run(shlex.split(cmd), check=True, stdout=PIPE).stdout.decode('utf-8').strip()
    trainid = "_W depend=afterok:{}".format(trainid)


  for decset in args.decodes:

    # decode
    input = os.path.join(args.data, "{}.src".format(decset))
    orig = os.path.join(args.data, "{}.src.orig".format(decset))
    tstmaster = os.path.join(args.data, "*.{}.*.xml.gz".format(decset))
    nums = '_'.join(args.model_nums)
    cmd="{decode} --qsubopts \"{trainid} _N {name}.{decset}.standalone.decode _j oe _o {stand}/{decset}.monitor\" --input_file {input} --trained_models {stand} --model_nums {nums} --num_best 1 --output_file {stand}/{decset}.decode".format(decode=args.decodecmd, trainid=trainid, name=args.name, decset=decset, stand=args.standalone, input=input, nums=nums)
    sys.stderr.write(cmd+"\n")
    decodeid = ""
    if args.do_standalone_decode:
      decodeid = run(shlex.split(cmd), check=True, stdout=PIPE).stdout.decode('utf-8').strip()
      decodeid = "-W depend=afterok:{}".format(decodeid)


    # package
    cmd= "qsubrun -N {name}.{decset}.standalone.package -j oe -o {stand}/{decset}.package.monitor {decodeid} -- {package} {stand}/{decset}.decode {orig} {tstmaster} {stand}/{name}-standalone.{lang}-eng.{decset}.y1r1.v2.xml.gz".format(package=args.packagecmd,  name=args.name, decset=decset, stand=args.standalone, decodeid=decodeid, orig=orig, tstmaster=tstmaster, lang=args.lang )
    if args.do_standalone_package:
      run(shlex.split(cmd), check=True)


def prepfile(fh, code):
  if type(fh) is str:
    fh = open(fh, code)
  ret = gzip.open(fh.name, code) if fh.name.endswith(".gz") else fh
  if sys.version_info[0] == 2:
    if code.startswith('r'):
      ret = reader(fh)
    elif code.startswith('w'):
      ret = writer(fh)
    else:
      sys.stderr.write("I didn't understand code "+code+"\n")
      sys.exit(1)
  return ret

def addonoffarg(parser, arg, dest=None, default=True, help="TODO"):
  ''' add the switches --arg and --no-arg that set parser.arg to true/false, respectively'''
  group = parser.add_mutually_exclusive_group()
  dest = arg if dest is None else dest
  group.add_argument('--%s' % arg, dest=dest, action='store_true', default=default, help=help)
  group.add_argument('--no-%s' % arg, dest=dest, action='store_false', default=not default, help="See --%s" % arg)

def main():
  parser = argparse.ArgumentParser(description="run training, decoding, packaging of standalone and parent model all together",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  addonoffarg(parser, 'debug', help="debug mode", default=False)

  parser.add_argument("--name", "-n", required=True, help="a name for this training")
  parser.add_argument("--lang", "-l", required=True, help="language of the training")
  parser.add_argument("--standalone", "-s", required=True, help="model location (this model will be created)")
  parser.add_argument("--parent", "-p", required=True, help="model location (this model will be created)")
  parser.add_argument("--child", "-c", required=True, help="model location (this model will be created)")
  parser.add_argument("--epochs", "-e", type=int, default=40, help="number of epochs to run (child/standalone)")
  parser.add_argument("--parentepochs", "-P", type=int, default=1, help="number of epochs to run (parent)")
  parser.add_argument("--model_nums", "-N", type=str, nargs='+', default=[str(x) for x in range(1,9)], choices=[str(x) for x in range(1,9)], help="model variants to train")
  parser.add_argument("--data", required=True, help="path to data directory")
  parser.add_argument("--parent_data", required=True, help="path to parent data directory")
  parser.add_argument("--train_source", default="train.src", help="name of source side of (stand/child) training data")
  parser.add_argument("--train_target", default="train.trg",  help="target side of (stand/child) training data")
  parser.add_argument("--parent_source", default="train.src",  help="source side of parent data")
  parser.add_argument("--parent_target", default="train.trg",  help="target side of parent data")

  parser.add_argument("--dev_source", default="dev.src", help="source side of (stand/child) dev data")
  parser.add_argument("--dev_target", default="dev.trg", help="target side of (stand/child) dev data")

  parser.add_argument("--parent_dev_source", default="dev.src", help="source side of parent dev data")
  parser.add_argument("--parent_dev_target", default="dev.trg", help="target side of parent dev data")

  parser.add_argument("--decodes", nargs='+', default=["dev", "test", "syscomb", "eval"], help="sets to decode")
  parser.add_argument("--qsubopts", default="", help="additional options to pass to qsub")
  parser.add_argument("--extra_rnn_args", default="", help="additional options to pass to rnn binary")
  parser.add_argument("--traincmd", default=os.path.join(scriptdir, 'train_models.py'), help="training script")
  parser.add_argument("--decodecmd", default=os.path.join(scriptdir, 'decode_models.sh'), help="decode script")
  parser.add_argument("--packagecmd", default=os.path.join(scriptdir, 'packagenmt.sh'), help="package script")
  addonoffarg(parser, 'do_standalone', help="do standalone anything", default=True)
  addonoffarg(parser, 'do_standalone_train', help="do standalone training", default=True)
  addonoffarg(parser, 'do_standalone_decode', help="do standalone decoding", default=True)
  addonoffarg(parser, 'do_standalone_package', help="do standalone packaging", default=True)
  addonoffarg(parser, 'do_parent', help="do parent anything", default=True)
  addonoffarg(parser, 'do_child', help="do child anything", default=True)
  addonoffarg(parser, 'do_child_train', help="do child training", default=True)
  addonoffarg(parser, 'do_child_decode', help="do child decoding", default=True)



  workdir = tempfile.mkdtemp(prefix=os.path.basename(__file__), dir=os.getenv('TMPDIR', '/tmp'))

  try:
    args = parser.parse_args()
  except IOError as msg:
    parser.error(str(msg))

  def cleanwork():
    shutil.rmtree(workdir, ignore_errors=True)
  if args.debug:
    print(workdir)
  else:
    atexit.register(cleanwork)

  jobids = []

  # standalone
  if args.do_standalone:
    _standalone(args)
  # TODO: continue on with parent and child?

if __name__ == '__main__':
  main()
