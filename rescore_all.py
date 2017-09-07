#!/usr/bin/env python3
# code by Jon May [jonmay@isi.edu], Leon Cheung [lcheung@isi.edu]
import argparse
import sys
import codecs
if sys.version_info[0] == 2:
  from itertools import izip
else:
  izip = zip
from collections import defaultdict as dd
import re
import os.path
import gzip
import tempfile
import shutil
import atexit
from jmutil import mkdir_p
from subprocess import check_output, check_call
import shlex
scriptdir = os.path.dirname(os.path.abspath(__file__))

reader = codecs.getreader('utf8')
writer = codecs.getwriter('utf8')

JOBS=set()
def cleanjobs():
  for job in JOBS:
    if check_call(shlex.split("qdel {}".format(job))) != 0:
      sys.stderr.write("Couldn't delete {}\n".format(job))
atexit.register(cleanjobs)


def _rerankmodel(devadj, devid, rerankweights, outfile, args):
  ''' run rerank model: gets weights for reranking '''
  # figure out what feature list is actually going to be (easier to just make it than catch output of combineid)
  nmts = []
  if args.model:
    nmts = [ 'nmt_%d' % x for x in range(len(args.model_nums)) ]
  if args.model_dir is not None: # signal for dl4mt model in use
    nmts.append('nmt_c2c')
  if args.tf_script:
    nmts.append('nmt_tf')
  featels = ' '.join(nmts)
  print('features;', featels)
  # run actual rerank
  print(devadj)
  tunererank="{}/{}.{}".format(args.root, os.path.basename(devadj), args.suffix)
  
  oldfeats="text-length derivation-size lm1 lm2"
  cmd="qsubrun -j oe -o {root}/rerankmodel.monitor -N {label}.rerankmodel -W depend=afterok:{devid} -- {rerankmodel} -f {oldfeats} {featels} -w {weights} -r {devref} -o {rerankweights} -i {devadj} -b {tunererank}".format(tunererank=tunererank, root=args.root, devid=devid, rerankmodel=os.path.join(args.pipeline, args.rerankmodel), oldfeats=oldfeats, featels=featels, weights=os.path.join(args.input, "weights.final"), devref=os.path.join(args.input, "{}.trg.ref".format(args.dev)), devadj=devadj, label=args.label, rerankweights=rerankweights)
  outfile.write(cmd+"\n")
  job = check_output(shlex.split(cmd)).decode('utf-8').strip()
  JOBS.add(job)
  outfile.write(job+"\n")
  return job

def _applyrerank(corpus, adjoin, idstr, rerankweights, decodefile, outfile, args):
  ''' apply rerank model'''
  cmd="qsubrun -j oe -o {root}/rerank.{corpus}.monitor -N {corpus}.{label}.rerank -W depend=afterok:{idstr} -- {applymodel} -i {adjoin} -w {tuneweights} -k {rerankweights} -b {decodefile}".format(root=args.root, idstr=idstr, applymodel=os.path.join(args.pipeline, args.rerankapply), adjoin=adjoin, tuneweights=os.path.join(args.input, "weights.final"), rerankweights=rerankweights, decodefile=decodefile, corpus=corpus, label=args.label)
  outfile.write(cmd+"\n")
  job = check_output(shlex.split(cmd)).decode('utf-8').strip()
  JOBS.add(job)
  outfile.write(job+"\n")
  return job

def _package(corpus, idstr, orig, tstmaster, decodefile, outfile, args):
  ''' make elisa package '''
  cmd= "qsubrun -N {corpus}.{label}.package -j oe -o {root}/{corpus}.package.monitor -W depend=afterok:{idstr} -- {package} {decodefile} {orig} {tstmaster} {root}/{label}-rescore.{lang}-eng.{corpus}.y1r1.v2.xml.gz".format(root=args.root, package=args.packagecmd, decodefile=decodefile, corpus=corpus, orig=orig, tstmaster=tstmaster,  label=args.label, lang=args.lang, idstr=idstr )
  outfile.write(cmd+"\n")
  job = check_output(shlex.split(cmd)).decode('utf-8').strip()
  JOBS.add(job)
  outfile.write(job+"\n")


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
  parser = argparse.ArgumentParser(description="hpc launch to rescore n-best lists with a given model",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  addonoffarg(parser, 'debug', help="debug mode", default=False)
  parser.add_argument("--input", "-i", help="input directory containing *.src.hyp, *.trg.ref, weights.final for each set for a language")
  parser.add_argument("--outfile", "-o", nargs='?', type=argparse.FileType('w'), default=sys.stdout, help="output file")
  parser.add_argument("--model", "-m", default=None, help="path to zoph trained model")
  parser.add_argument("--model_nums", "-n", nargs='+', type=int, default=[1,2,3,4,5,6,7,8], help="which models to use")
  parser.add_argument("--dev", "-d", type=str, default="dev", help="set to optimize on")
  parser.add_argument("--lang", "-L", required=True, help="language of the training")
  parser.add_argument("--label", "-l", type=str, default="x", help="label for job names")
  parser.add_argument("--eval", "-e", nargs='+', type=str, default=["dev", "test", "syscomb"], help="sets to evaluate on")
  parser.add_argument("--root", "-r", help="path to put outputs")
  parser.add_argument("--qsubopts", default=None, help="additional options to pass to qsub")
  parser.add_argument("--suffix", "-S", help="goes on the end of final onebest", default="onebest.rerank")
  parser.add_argument("--rescore_single", default=os.path.join(scriptdir, "rescore_split.py"), help="rescore script")
  parser.add_argument("--rescore_single_dl", default=None, help="rescore script")
  parser.add_argument("--rescore_split_dl", default=os.path.join(scriptdir, "rescore_split_dl.py"), help="rescore script")
  parser.add_argument("--convert", default=os.path.join(scriptdir, "nmtrescore2sbmtnbest.py"), help="adjoin scores")
  parser.add_argument("--pipeline", default='/home/nlg-02/pust/pipeline-2.22', help="sbmt pipeline")
  parser.add_argument("--runrerank", default='runrerank.sh', help="runrerank script")
  parser.add_argument("--rerankmodel", default=os.path.join('scripts', 'runrerank.py'), help="inner runrerank model script")
  parser.add_argument("--rerankapply", default=os.path.join('scripts', 'applyrerank.py'), help="inner runrerank apply script")
  parser.add_argument("--packagecmd", default=os.path.join(scriptdir, 'packagenmt.sh'), help="package script")
  parser.add_argument("--model_dir", default=None, help="dir of grads")
  parser.add_argument("--model_pkl", default="bi-char2char.pkl", help="pkl file")
  parser.add_argument("--model_grads", default="bi-char2char.grads.rec.npz", help="grads.[.+].npz")
  parser.add_argument("--dict_src", default=None, help="path to src dict")
  parser.add_argument("--dict_trg", default=None, help="path to trg dict")
  parser.add_argument("--n_words_src", default=150, help="src dict size")
  parser.add_argument("--n_words", default=440, help="trg dict size")
  parser.add_argument("--width", "-w", type=int, default=5, help="num rescore job processes")
  parser.add_argument("--tf-script", default=None, help="path to shell script to run tensorflow forced decoding")
  parser.add_argument("--rescore_split_tf", default=os.path.join(scriptdir, "rescore_split_tf.py"), help="rescore split script for tf")
  addonoffarg(parser, 'skipnmt', help="assume nmt results already exist and skip them", default=False)

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

  outfile = prepfile(args.outfile, 'w')
  mkdir_p(args.root)
  datasets = set(args.eval)
  datasets.add(args.dev)

  combineids = {}
  adjoins = {}
  qsub = ""
  if args.qsubopts is not None:
    qsub = "--extra_qsub_opts=\"{}\"".format(args.qsubopts)
  global JOBS
  for dataset in datasets:
    jobids = []
    allscores = []

    # rescore submissions; catch jobids
    data = os.path.realpath(os.path.join(args.input, "{}.src.hyp".format(dataset)))
    if not os.path.exists(data):
      sys.stderr.write("ERROR: {} does not exist\n".format(data))
      sys.exit(1)

    # zoph support
    if args.model is not None:
      modelroot=os.path.realpath(args.model)
      if not os.path.exists(modelroot): #TODO: also check model number!
        sys.stderr.write("ERROR: {} does not exist\n".format(modelroot))
        sys.exit(1)

      for model in args.model_nums:
        scores = os.path.realpath(os.path.join(args.root, "{}.m{}.scores".format(dataset, model)))
        allscores.append(scores)
        if args.skipnmt:
          if not os.path.exists(scores):
            sys.stderr.write("ERROR: Skipping nmt but {} does not exist\n".format(scores))
            sys.exit(1)
        else:
          split_workdir = os.path.abspath(os.path.join(args.root, '%s-zoph' % dataset))
          sys.stderr.write('Using %s as splitting workdir' % split_workdir)
          log = os.path.realpath(os.path.join(args.root, "{}.m{}.log".format(dataset, model)))
          cmd = "{rescore} {qsub} --workdir {split_workdir} --splitsize {width} --model {modelroot} --modelnum {model} --datafile {data} --outfile {scores} --logfile {log}".format(qsub=qsub, model=model, rescore=args.rescore_single, modelroot=modelroot, data=data, split_workdir=split_workdir, scores=scores, width=args.width, log=log)
          outfile.write(cmd+"\n")
          job = check_output(shlex.split(cmd)).decode('utf-8').strip()
          JOBS.add(job)
          jobids.append(job)

    # dl4mt support
    if args.model_dir is not None:
      scores = os.path.realpath(os.path.join(args.root, "{}.dl.scores".format(dataset)))
      allscores.append(scores)
      if args.skipnmt:
        if not os.path.exists(scores):
          sys.stderr.write("ERROR: Skipping nmt but {} does not exist\n".format(scores))
          sys.exit(1)
      else:
        log = os.path.realpath(os.path.join(args.root, "{}-rescore.log".format(dataset)))
        split_workdir = os.path.abspath(os.path.join(args.root, '%s-dl' % dataset))
        sys.stderr.write('Using %s as splitting workdir' % split_workdir)
        cmd = ("{rescore} {qsub}"
                 " --rescore_single {rescore_single_dl}"
                 " --splitsize {splitsize}"
                 " --workdir {split_workdir}"
                 " --model_dir {model_dir}"
                 " --model_pkl {model_pkl}"
                 " --model_grads {model_grads}"
                 " --outfile {outfile}"
                 " --dict_src {dict_src} --dict_trg {dict_trg}"
                 " --n_words_src {n_words_src} --n_words {n_words}"
                 " --datafile {datafile}"
                 " --logfile {log}").format(
                 rescore=args.rescore_split_dl, qsub=qsub,
                    rescore_single_dl=args.rescore_single_dl,
                    splitsize=args.width,
                    split_workdir=split_workdir,
                    model_dir=args.model_dir,
                    model_pkl=args.model_pkl,
                    model_grads=args.model_grads,
                    outfile=scores,
                    dict_src=args.dict_src, dict_trg=args.dict_trg,
                    n_words_src=args.n_words_src, n_words=args.n_words,
                    datafile=data,
                    log=log)
        outfile.write(cmd+"\n")
        job = check_output(shlex.split(cmd)).decode('utf-8').strip()
        assert os.path.isdir(split_workdir)
        JOBS.add(job)
        jobids.append(job)

    if args.tf_script is not None:
      scores = os.path.realpath(os.path.join(args.root, "{}.tf.scores".format(dataset)))
      allscores.append(scores)
      if args.skipnmt:
        if not os.path.exists(scores):
          sys.stderr.write("ERROR: Skipping nmt but {} does not exist\n".format(scores))
          sys.exit(1)
      else:
        split_workdir = os.path.abspath(os.path.join(args.root, '%s-tf' % dataset))
        sys.stderr.write('Using %s as splitting workdir.\n' % split_workdir)
        log = os.path.realpath(os.path.join(args.root, "{}-rescore.log".format(dataset)))
        cmd = ("{rescore} {qsub}"
                 " --rescore_single {tf_script}"
                 " --splitsize {splitsize}"
                 " --workdir {split_workdir}"
                 " --datafile {datafile}"
                 " --outfile {outfile}"
                 " --log {log}").format(
                 rescore=args.rescore_split_tf, qsub=qsub,
                    tf_script=args.tf_script,
                    splitsize=args.width,
                    split_workdir=split_workdir,
                    datafile=data,
                    outfile=scores,
                    log=log)
        outfile.write(cmd+"\n")
        job = check_output(shlex.split(cmd)).decode('utf-8').strip()
        JOBS.add(job)
        jobids.append(job)
      
    # combine rescores and paste in previous nbests; (for each dataset)
    jobidstr = "-W depend=afterok:"+':'.join(jobids) if len(jobids)>=1 else ""
    scorestr = ' '.join(allscores)
    print ('Scores;', scorestr)
    nbest = os.path.join(args.input, "{}.nbest".format(dataset))
    adjoin = os.path.join(args.root, "{}.adjoin.{}".format(dataset, args.suffix))
    adjoins[dataset] = adjoin
    cmd = "qsubrun -j oe -o {root}/{dataset}.convert.monitor -N {label}.{dataset}.convert {jobidstr} -- {convert} -i {scorestr} -a {nbest} -o {adjoin}".format(root=args.root, dataset=dataset, jobidstr=jobidstr, convert=args.convert, scorestr=scorestr, nbest=nbest, adjoin=adjoin, label=args.label)
    outfile.write(cmd+"\n")
    job = check_output(shlex.split(cmd)).decode('utf-8').strip()
    JOBS.add(job)
    combineids[dataset] = job

  # the rerank model
  rerankweights="{}/rerank.weights".format(args.root)
  modeljob = _rerankmodel(adjoins[args.dev], combineids[args.dev], rerankweights, outfile, args)

  # apply and package
  for dataset in datasets:
    orig=os.path.join(args.input, "{}.src.orig".format(dataset))
    tstmaster = os.path.join(args.input, "*.{}.*.xml.gz".format(dataset))
    decodefile = "{}/{}.decode".format(args.root, dataset)
    applyid = _applyrerank(dataset, adjoins[dataset], ':'.join([modeljob, combineids[dataset]]), rerankweights, decodefile, outfile, args)
    _package(dataset, applyid, orig, tstmaster, decodefile, outfile, args)

  # (TODO: run bleu)

  # no more atexit job deletion
  JOBS = []
if __name__ == '__main__':
  main()
