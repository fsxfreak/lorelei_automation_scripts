#!/usr/bin/env python3
# code by Jon May [jonmay@isi.edu]
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
import os
import gzip
import tempfile
import shutil
import atexit
import shlex
from jmutil import mkdir_p
from subprocess import run, Popen, PIPE
scriptdir = os.path.dirname(os.path.abspath(__file__))

reader = codecs.getreader('utf8')
writer = codecs.getwriter('utf8')


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
  parser = argparse.ArgumentParser(description="run rescoring/force decoding over the cluster",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  addonoffarg(parser, 'debug', help="debug mode", default=False)
  parser.add_argument("--splitsize", "-z", type=int, default=1, help="number of batches")
  parser.add_argument("--datafile", "-d", type=str, required=True, help="input source tab trg file")
  parser.add_argument("--model", "-m", required=True, help="model file")
  parser.add_argument("--modelnum", "-n", required=True, help="model number")
  parser.add_argument("--logfile", "-l", type=str, default='/dev/null', help="where to log data")
  parser.add_argument("--outfile", "-o", type=str, required=True, help="output scores file")
  parser.add_argument("--workdir", "-w", default=None, help="work directory (defaults to 'work' subdir of outfile")
  parser.add_argument("--extra_rnn_args", help="extra arguments to rnn binary")
  parser.add_argument("--extra_qsub_opts", "-q", help="extra options to qsubrun (scorers only)")
  parser.add_argument("--rescore_single", default=os.path.join(scriptdir, 'rescore_single.py'), help="rescore single script")
  parser.add_argument("--cat", default=os.path.join(scriptdir, 'cat.py'), help="cat with named output")
  parser.add_argument("--rnn_location", default=os.path.join(scriptdir, 'helper_programs', 'ZOPH_RNN'), help="rnn binary")


  try:
    args = parser.parse_args()
  except IOError as msg:
    parser.error(str(msg))


  workdir = os.path.join(os.path.dirname(args.outfile), 'work') if args.workdir is None else args.workdir
  mkdir_p(workdir)
  qsubopts = args.extra_qsub_opts if args.extra_qsub_opts is not None else ""

  # split into desired number of pieces
  splitcmd = run(shlex.split("split -a 1 -n l/{} -d {} {}/data.".format(args.splitsize, args.datafile, workdir)), check=True)
  jobids = []
  outfiles = []
  joincmd = "qsubrun -q isi -l walltime=24:00:00 -j oe -o {logfile} -N {outfile}.join -W depend=afterok:".format(logfile=args.logfile, outfile=args.outfile)
  for piece in range(args.splitsize):
    # split back into source and target
    df = "{}/data.{}".format(workdir, piece)
    of = "{}/scores.{}".format(workdir, piece)
    outfiles.append(of)
    # launch individual rescore jobs; collect job ids
    cmd="qsubrun -o {workdir}/split.log.{piece} -N split.{piece} {qsub} -- {rescore_single} -m {model} -n {modelnum} -d {df} -o {of} -l {workdir}/inner.log.{piece}".format(qsub=qsubopts, workdir=workdir, piece=piece, rescore_single=args.rescore_single, model=args.model, modelnum=args.modelnum, df=df, of=of)
    #print("Launching {}".format(cmd))
    jobid = run(shlex.split(cmd), stdout=PIPE).stdout.decode('utf-8').strip()
    #print("Got {}".format(jobid))
    jobids.append(jobid)
  joincmd += "{} -- {} -i {} -o {} ".format(':'.join(jobids), args.cat, ' '.join(outfiles), args.outfile)
  #print(joincmd)
  res = run(shlex.split(joincmd), stdout=PIPE).stdout.decode('utf-8').strip()
  print(res)
if __name__ == '__main__':
  main()
