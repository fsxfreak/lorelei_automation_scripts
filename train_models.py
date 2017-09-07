#!/usr/bin/env python3
# code by Jon May [jonmay@isi.edu]
# rewrite of zoph train_models that returns job ids, takes qsub cmds

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


def runaligner(args):
  ''' launch berkeley aligner '''
  root = args.trained_model
  alroot = os.path.join(root, 'berk_aligner')
  dataroot=os.path.join(alroot, 'data')
  for source, target, name in zip((args.train_source, args.dev_source), (args.train_target, args.dev_target), ('train', 'test')):
    datadir=os.path.join(dataroot, name)
    mkdir_p(datadir)
    shutil.copy(source, os.path.join(datadir, "{}.f".format(name)))
    shutil.copy(target, os.path.join(datadir, "{}.e".format(name)))
  for file in (args.aligncmd, args.alignjar, args.alignconf):
    shutil.copy(file, alroot)
  cmd="qsubrun -N {name}.align -j oe -o {alroot}/align.monitor {qsubopts} -- {berkalignsh} {alroot}".format(alroot=alroot, qsubopts=args.qsubopts, berkalignsh=args.berkalignsh, name=args.name)
  sys.stderr.write(cmd+"\n")
  jobid = run(shlex.split(cmd), check=True, stdout=PIPE).stdout.decode('utf-8').strip()
  return jobid

def runchildmodel(args, modelnum):
  ''' launch child model '''
  root = args.trained_model
  modelroot=os.path.join(root, "model{}".format(modelnum))
  parentroot = args.parent_model
  parentmodelroot=os.path.join(parentroot, "model{}".format(modelnum))

  cmd="qsubrun -N {name}.{mode}.{modelnum}.train -j oe -o {modelroot}/train.monitor {qsubopts} -- {pretrain} --parent {parentmodelroot} -ts {trainsource} -tt {traintarget} -ds {devsource} -dt {devtarget} -c {modelroot} -n {epochs} --logfile {modelroot}/train.log --rnnbinary {rnnbin} {extraargs}".format(mode=args.mode, parentmodelroot=parentmodelroot, name=args.name, modelnum=modelnum, pretrain=args.pretrain, modelroot=modelroot, qsubopts=args.qsubopts, trainsource=args.train_source, traintarget=args.train_target, devsource=args.dev_source, devtarget=args.dev_target, epochs=args.epochs, rnnbin=args.rnn_binary, extraargs=args.extra_rnn_args)
  sys.stderr.write(cmd+"\n")
  jobid = run(shlex.split(cmd), check=True, stdout=PIPE).stdout.decode('utf-8').strip()
  return jobid


def runmodel(args, modelnum, countfile, opts):
  ''' launch standalone/parent model (replaces train_single_model.sh) '''

  # start this many candidate models
  NUM_STARTS = 4
  starts = []
  if args.trained_model.endswith('/'):
    args.trained_model = args.trained_model[:-1]
  jobids = []
  for start in range(NUM_STARTS):
    root = '%s-%d' % (args.trained_model, start)
    modelroot=os.path.join(root, "model{}".format(modelnum))
    mkdir_p(modelroot)
    cmd=("qsubrun -N {name}.{mode}.{modelnum}.train"
        " -j oe -o {modelroot}/train.monitor {qsubopts}"
        " -- {rnnwrap} {rnnbin}"
        " -t {trainsource} {traintarget} {modelroot}/model.nn"
        " -B {modelroot}/best.nn"
        " -a {devsource} {devtarget} {opts}"
        " --vocab-mapping-file {countfile}"
        " --logfile {modelroot}/train.log").format(
            name=args.name, modelnum=modelnum, mode=args.mode,
            modelroot=modelroot, qsubopts=args.qsubopts, 
            rnnwrap=args.rnnwrap, rnnbin=args.rnn_binary,
            trainsource=args.train_source, traintarget=args.train_target, 
            devsource=args.dev_source, devtarget=args.dev_target, opts=opts, 
            countfile=countfile)
    sys.stderr.write(cmd+"\n")
    jobid = run(shlex.split(cmd), check=True, stdout=PIPE).stdout.decode('utf-8').strip()
    jobids.append(jobid)
    starts.append('%s|||%s' % (jobid, os.path.join(modelroot, 'train.log')))

  return ':'.join(jobids), starts

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
  MODES = ['parent', 'child', 'standalone']
  parser = argparse.ArgumentParser(description="train seq2seq standalone, parent, and child models",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  addonoffarg(parser, 'debug', help="debug mode", default=False)

  parser.add_argument("--name", required=True, help="a name for this training (combined with mode)")
  parser.add_argument("--mode", required=True, choices=MODES, help="what kind of training are we doing?")
  parser.add_argument("--trained_model", "-m", required=True, help="model location (this model will be created)")
  parser.add_argument("--parent_model", "-p", default=None, help="parent model location (if this model exists, it's a child model)")
  parser.add_argument("--model_nums", "-n", type=int, nargs='+', default=[x for x in range(1,9)], choices=range(1,9), help="model variants to train")
  parser.add_argument("--train_source", "-ts", required=True, help="source side of training data")
  parser.add_argument("--train_target", "-tt", required=True, help="target side of training data")
  parser.add_argument("--mapping_source", "-ms",  help="source side of child data when building parent (for mapping)")
  parser.add_argument("--mapping_target", "-mt",  help="target side of child data when building parent (for mapping)")
  parser.add_argument("--vocab_force", "-vf",  default=None, help="force target vocabulary when building parent")

  parser.add_argument("--dev_source", "-ds", required=True, help="source side of dev data")
  parser.add_argument("--dev_target", "-dt", required=True, help="target side of dev data")
  parser.add_argument("--epochs", "-e", type=int, default=40, help="number of epochs to run")
  parser.add_argument("--qsubopts", default="", help="additional options to pass to qsub")
  parser.add_argument("--extra_rnn_args", default="", help="additional options to pass to rnn binary")
  parser.add_argument("--outfile", "-o", nargs='?', type=argparse.FileType('w'), default=sys.stdout, help="output file")
  parser.add_argument("--previous_alignment", default=None, help="path to berk_align directory that will be simlinked in if align is false and mode is not parent")
  addonoffarg(parser, 'align', help="run aligner (never run when training parent)", default=True)
  parser.add_argument("--aligncmd", default=os.path.join(scriptdir, 'helper_programs', 'align'), help="aligner data")
  parser.add_argument("--alignjar", default=os.path.join(scriptdir, 'helper_programs', 'berkeleyaligner.jar'), help="aligner data")
  parser.add_argument("--alignconf",default=os.path.join(scriptdir, 'helper_programs', 'unk_replace.conf'), help="aligner data")
  parser.add_argument("--mappingstandalone",default=os.path.join(scriptdir, 'helper_programs', 'create_mapping_pureNMT.py'), help="mapping program")
  parser.add_argument("--mappingparent",default=os.path.join(scriptdir, 'helper_programs', 'create_mapping_parent.py'), help="mapping program")
  parser.add_argument("--mappingparentagn",default=os.path.join(scriptdir, 'helper_programs', 'create_mapping_parent_agnostic.py'), help="mapping program")
  parser.add_argument("--berkalignsh", default=os.path.join(scriptdir, 'helper_programs', 'berk_align.sh'), help="aligner cmd")
  parser.add_argument("--rnnwrap", default=os.path.join(scriptdir, 'helper_programs', 'rnn_wrap.sh'), help="rnn wrapper")
  parser.add_argument("--pretrain", default=os.path.join(scriptdir, 'helper_programs', 'pretrain.py'), help="pretrain child model trainer")
  parser.add_argument("--rnn_binary", default=os.path.join(scriptdir, 'helper_programs', 'ZOPH_RNN'), help="rnn binary")

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


  # model configuration

  dropouts = {}
  if args.mode == 'parent':
    args.align = False
    dropouts[1]="-d 0.8"
    dropouts[2]=""
  else:
    dropouts[1]=dropouts[2]="-d 0.5"

  modelopts={}
  for i in range(1, 9):
    H = 750 if int((i+1)/2) % 2 == 1 else 1000 # 1, 2, 5, 6 = 750
    N = 2 if (i % 2) == 1 else 3 # 1 3 5 7 = 2
    G = "0 1 1" if N == 2 else "0 0 1 1" # 1 3 5 7 = 0 0 1
    dr = dropouts[1] if i <=4 else dropouts[2]
    modelopts[i] = "-m 128 -l 0.5 -P -0.08 0.08 -w 5 --attention-model 1 --feed-input 1 --screen-print-rate 30  -n {epochs} -L 100 {dr} -H {H} -N {N} -M {G}".format(epochs=args.epochs, H=H, N=N, G=G, dr=dr)
    #sys.stderr.write("Model {}: {}\n".format(i, modelopts[i]))

  if not ((args.mode == 'child') ^ (args.parent_model is None)):
    sys.stderr.write("Should only (and always) specify parent model when building child\n")
    sys.exit(1)

  mkdir_p(args.trained_model)

  # run aligner
  if args.align:
    jobids.append(runaligner(args))
  elif args.mode != 'parent' and args.previous_alignment is not None:
    dst = os.path.join(args.trained_model, 'berk_aligner')
    if not os.path.exists(dst):
      os.symlink(os.path.abspath(args.previous_alignment), dst)

  # if not child model, obtain vocabulary based on token frequency
  if args.mode == 'child':
    pass
  else:
    if args.mode == 'standalone':
      cmd = "{mapping} {trainsource} {traintarget} 6 {modelroot}/count6.nn".format(mapping=args.mappingstandalone, trainsource=args.train_source, traintarget=args.train_target, modelroot=args.trained_model)
    elif args.mode == 'parent' and not args.vocab_force:
      cmd = "{mapping} {mapsource} {maptarget} 6 {modelroot}/count6.nn {trainsource}".format(mapping=args.mappingparent, mapsource=args.mapping_source, maptarget=args.mapping_target, trainsource=args.train_source, modelroot=args.trained_model)
    elif args.mode == 'parent' and args.vocab_force:
      cmd = "{mapping} {mapsource} {maptarget} 6 {modelroot}/count6.nn {trainsource}".format(mapping=args.mappingparentagn, mapsource=args.mapping_source, maptarget=args.vocab_force, trainsource=args.train_source, modelroot=args.trained_model)
    sys.stderr.write(cmd+"\n")
    run(shlex.split(cmd), check=True)

  # launch trainings
  for modelnum in args.model_nums:
    modelpath = os.path.join(args.trained_model, "model{}".format(modelnum))
    mkdir_p(modelpath)
    if args.mode == 'child':
      jobids.append(runchildmodel(args, modelnum))
    else:
      countfile = "{}/count6.nn".format(args.trained_model)
      #opts=modelopts[modelnum]+" -B {}/best.nn".format(modelpath)
      opts=modelopts[modelnum]
      jobids_out, starts = runmodel(args, modelnum, countfile, opts)
      jobids.append(jobids_out)

      watch_cmd = ['qsubrun', '-j', 'oe', '-o', '/home/nlg-05/ljcheung/logs', 
          '--', os.path.join(scriptdir, 'watch_training.py'), 
          '--watches', ':'.join(starts),
          '--final_model_dir', args.trained_model]
      watchid = run(watch_cmd, 
          check=True, stdout=PIPE).stdout.decode('utf-8').strip()
      jobids.append(watchid)
      sys.stderr.write('%s\n' % ' '.join(watch_cmd))
 
  outfile = prepfile(args.outfile, 'w')
  outfile.write('%s\n' % ':'.join(jobids))

if __name__ == '__main__':
  main()
