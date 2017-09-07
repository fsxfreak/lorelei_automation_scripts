#!/usr/bin/env python3
# code adapted by Leon Cheung [lcheung@isi.edu] from Jon May [jonmay@isi.edu]
# specialized for tf-seq2seq model
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
  parser.add_argument("--logfile", "-l", type=str, default='/dev/null', help="where to log data")
  parser.add_argument("--outfile", "-o", type=str, required=True, help="output scores file")
  parser.add_argument("--workdir", "-w", default=None, help="work directory (defaults to 'work' subdir of outfile")
  parser.add_argument("--extra_rnn_args", help="extra arguments to rnn binary")
  parser.add_argument("--extra_qsub_opts", "-q", help="extra options to qsubrun (scorers only)")
  parser.add_argument("--rescore_single", required=True, help="rescore single script")
  parser.add_argument("--cat", default=os.path.join(scriptdir, 'cat.py'), help="cat with named output")

  try:
    args = parser.parse_args()
  except IOError as msg:
    parser.error(str(msg))

  workdir = os.path.join(os.path.dirname(args.outfile), 'work') if args.workdir is None else args.workdir
  mkdir_p(workdir)
  qsubopts = args.extra_qsub_opts if args.extra_qsub_opts is not None else ""

  # split into desired number of pieces
  fill = len(str(args.splitsize-1))
  splitcmd = run(shlex.split("split -a {} -n l/{} -d {} {}/data.".format(fill, args.splitsize, args.datafile, workdir)), check=True)
  jobids = []
  outfiles = []
  joincmd = "qsubrun -q isi -l walltime=24:00:00 -j oe -o {logfile} -N {outfile}.join -W depend=afterok:".format(logfile=args.logfile, outfile=args.outfile)
  for piece in range(args.splitsize):
    # split back into source and target
    piece = str(piece).zfill(fill)
    df = "{}/data.{}".format(workdir, piece)
    of = "{}/scores.{}".format(workdir, piece)
    outfiles.append(of)
    # launch individual rescore jobs; collect job ids

    cmd=("qsubrun -o {workdir}/split.log.{piece}"
                " -N spt.{piece}"
                " {qsubopts}"
                " -- {rescore_single}"
                " {df} {of}"
                " {workdir}").format(
                  workdir=workdir, piece=piece,
                  qsubopts=qsubopts,
                  rescore_single=args.rescore_single,
                  of=of, df=df)


    sys.stderr.write("Launching {}".format(cmd)+"\n")
    jobid = run(shlex.split(cmd), stdout=PIPE).stdout.decode('utf-8').strip()
    sys.stderr.write("Got {}".format(jobid)+"\n")
    jobids.append(jobid)
  joincmd += "{} -- {} -i {} -o {} ".format(':'.join(jobids), args.cat, ' '.join(outfiles), args.outfile)
  sys.stderr.write(joincmd+"\n")
  res = run(shlex.split(joincmd), stdout=PIPE).stdout.decode('utf-8').strip()
  print(res)
if __name__ == '__main__':
  main()

