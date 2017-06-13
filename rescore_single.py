#!/usr/bin/env python3
# code by Jon May [jonmay@isi.edu]
#PBS -q isi
#PBS -l walltime=96:00:00
#PBS -l gpus=2
#PBS -j oe

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

def getgpucount(default=1):
  env = os.environ
  ret = default
  if 'PBS_GPUFILE' in env:
    cmd="wc -l {}".format(env['PBS_GPUFILE'])
    print("Cmd is {}".format(cmd))
    ret=int(run(shlex.split(cmd), check=True, stdout=PIPE).stdout.decode('utf-8').split()[0])
  print("Got {} for getgpucount".format(ret))
  return ret

def getlongest(filea, fileb):
  ''' use wc -L to get length of the longest file in filea, fileb '''
  # TODO: could generalize number of args!
  la = int(run(shlex.split("wc -L {}".format(filea)), check=True, stdout=PIPE).stdout.decode('utf-8').split()[0])
  lb = int(run(shlex.split("wc -L {}".format(fileb)), check=True, stdout=PIPE).stdout.decode('utf-8').split()[0])
  return max(la, lb)
  
def main():
  parser = argparse.ArgumentParser(description="run rescoring/force decoding; take advantage of multi gpu",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  addonoffarg(parser, 'debug', help="debug mode", default=False)
  parser.add_argument("--datafile", "-d", type=str, required=True, help="input datafile")
  parser.add_argument("--model", "-m", required=True, help="model file")
  parser.add_argument("--modelnum", "-n", required=True, help="model number")
  parser.add_argument("--logfile", "-l", type=str, default=None, help="where to log data")
  parser.add_argument("--outfile", "-o", nargs='?', type=argparse.FileType('w'), default=sys.stdout, help="output scores file")
  parser.add_argument("--extra_rnn_args", help="extra arguments to rnn binary")
  parser.add_argument("--rnn_location", default=os.path.join(scriptdir, 'helper_programs', 'ZOPH_RNN'), help="rnn binary")



  try:
    args = parser.parse_args()
  except IOError as msg:
    parser.error(str(msg))
  go(args)

def go(args):
  workdir = tempfile.mkdtemp(prefix=os.path.basename(__file__), dir=os.getenv('TMPDIR', '/tmp'))
  def cleanwork():
    shutil.rmtree(workdir, ignore_errors=True)
  if args.debug:
    print(workdir)
  else:
    atexit.register(cleanwork)

  outfile = prepfile(args.outfile, 'w')
 # figure out how many gpus there are
  # split data accordingly
  # for each split,
  #   set gpu
  #   launch job
 
  gpus=getgpucount(default=2)
  splitcmd = run(shlex.split("split -a 1 -n l/{gpus} -d {name} {workdir}/data.".format(gpus=gpus, name=args.datafile, workdir=workdir)), check=True)
  procs = []
  for gpu in range(gpus):
    sf = prepfile("{}/source.{}".format(workdir, gpu), 'w')
    tf = prepfile("{}/target.{}".format(workdir, gpu), 'w')
    
    run(shlex.split("cut -f 1 {}/data.{}".format(workdir, gpu)), stdout=sf)
    run(shlex.split("cut -f 2 {}/data.{}".format(workdir, gpu)), stdout=tf)
    sf.close()
    tf.close()
    sf = sf.name
    tf = tf.name
    subworkdir = tempfile.mkdtemp(prefix="gpu", dir=workdir)
    print(sf, tf)
    longest = getlongest(sf, tf)
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES']="{}".format(gpu)
    output = "{}/output".format(subworkdir)
    print(output)
    if args.logfile is not None:
      logfile = "--logfile {}.{}".format(args.logfile, gpu)
    else:
      logfile = ""
    cmd = "{bin} -f {src} {trg} {model}/model{mnum}/best.nn {output} -L {longest} -m 1 {logfile} --tmp-dir-location {subwork}".format(bin=args.rnn_location, src=sf, trg=tf, model=args.model, mnum=args.modelnum, output=output, longest=longest, logfile=logfile, gpu=gpu, subwork=subworkdir)
    print("Running {}".format(cmd))
    procs.append((cmd, Popen(shlex.split(cmd), env=env), output))
  for cmd, proc, output in procs:
    print("Waiting for {}".format(cmd))
    proc.wait()
    print("Done with {}".format(cmd))
    for line in prepfile(output, 'r'):
      outfile.write(line)
if __name__ == '__main__':
  main()
