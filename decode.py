#!/usr/bin/env python3
# code by Jon May [jonmay@isi.edu]
#PBS -q isi
#PBS -l walltime=24:00:00
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
from itertools import cycle
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
    #print("Cmd is {}".format(cmd))
    ret=int(run(shlex.split(cmd), check=True, stdout=PIPE).stdout.decode('utf-8').split()[0])
  #print("Got {} for getgpucount".format(ret))
  return ret

MODELSEQ = [1,5,2,6,3,7,4,8]

def get_model_config(args):
  ''' get multi gpu options and model sequence '''
  mflags="-M"
  modseq=""
  gpus = cycle(range(getgpucount()))
  for model in MODELSEQ:
    if model in args.modelnum:
      # support ensemble of models
      gpu_allocations = [ '%s' % next(gpus) for _ in args.model ]
      mflags += ' '.join(gpu_allocations)
      #mflags+=" {}".format(next(gpus))

      best_nns = [ os.path.join(e, 'model%s' % model, 'best.nn') 
                   for e in args.model ]
      modseq += ' '.join(best_nns)
      #modseq+=" {}".format(os.path.join(args.model, "model{}".format(model), "best.nn"))
  return (mflags, modseq)
      
def prepare_data(args, workdir):
  ''' copy source data for models and make proper argument '''
  fseq = ""
  for model in args.modelnum:
    fname=os.path.join(workdir, "src.{}.txt".format(model))
    shutil.copy(args.input, fname)
    
    # copy input file multiple times for the ensemble of models
    fseqs = [ '%s' % fname for _ in args.model ]
    fseq += ' %s' % ' '.join(fseqs)
  return(fseq)
  

def getlongest(*files):
  ''' get length in words of the longest file '''
  max = 0
  for file in files:
    for line in prepfile(file, 'r'):
      wc = len(line.split())
      if wc > max:
        max = wc
  return max

def main():
  parser = argparse.ArgumentParser(description="run nmt decoding",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  addonoffarg(parser, 'debug', help="debug mode", default=False)
  parser.add_argument("--input", "-i", type=str, required=True, help="input datafile")
  parser.add_argument("--model", "-m", nargs='+', required=True, help="model file")
  parser.add_argument("--modelnum", "-n", nargs='+', type=int, default=[x for x in range(1,9)], help="model numbers")
  parser.add_argument("--logfile", "-l", type=str, default=None, help="where to log data")
  parser.add_argument("--outfile", "-o", type=str, required=True, help="output translations file")
  parser.add_argument("--extra_rnn_args", help="extra arguments to rnn binary")
  parser.add_argument("--rnn_location", default=os.path.join(scriptdir, 'helper_programs', 'ZOPH_RNN'), help="rnn binary")


  parser.add_argument("--bleu_format", default=os.path.join(scriptdir, 'helper_programs', "bleu_format.py"  ), help='conversion script')
  parser.add_argument("--att_unk_rep", default=os.path.join(scriptdir, 'helper_programs', "att_unk_rep.py"  ), help='conversion script')
  parser.add_argument("--decode_format", default=os.path.join(scriptdir, 'helper_programs', "decode_format.py"), help='conversion script')
  parser.add_argument("--ttable", default=os.path.join("berk_aligner","aligner_output","stage2.2.params.txt"), help='ttable file; assumed to be in model directory')

  try:
    args = parser.parse_args()

    args = parser.parse_args()
  except IOError as msg:
    parser.error(str(msg))


  ttable = os.path.join(args.model[0], args.ttable)
  workdir = tempfile.mkdtemp(prefix=os.path.basename(__file__), dir=os.getenv('TMPDIR', '/tmp'))
  def cleanwork():
    shutil.rmtree(workdir, ignore_errors=True)
  if args.debug:
    print(workdir)
  else:
    atexit.register(cleanwork)

  longest = getlongest(args.input)
  fseq = prepare_data(args, workdir)
  # FINAL_ARGS="\" $RNN_LOCATION --logfile $LOG_FILE -k $KBEST_SIZE $MODEL_NAMES $OUTPUT_FILE -b $BEAM_SIZE --print-score 1 $MFLAGS -L $LONGEST_SENT $EXTRA_RNN_ARGS \""
  mflags, modseq = get_model_config(args)
  unks = os.path.join(workdir, 'unks.txt')
  outtmp = os.path.join(workdir, 'outtmp.txt')
  cmd = "{zophrnn} --logfile {logfile} -k {kbest} {modseq} {outfile} -b {beam} --print-score 1 {mflags} -L {longest} {extraargs} --decode-main-data-files {fseq} --UNK-decode {unks} --tmp-dir-location {workdir}".format(zophrnn=args.rnn_location, logfile=args.logfile, kbest=1, modseq=modseq, outfile=args.outfile, beam=12, mflags=mflags, longest=longest, extraargs=args.extra_rnn_args, fseq=fseq, unks=unks, workdir=workdir)
  sys.stderr.write(cmd+"\n")
  run(shlex.split(cmd), check=True)
  # set up temporary copies of files
  shutil.copy(args.outfile, outtmp)
#  shutil.copy(args.outfile, args.outfile+".postout")
  cmd="{bleuformat} {outfile}".format(bleuformat=args.bleu_format, outfile=args.outfile)
  sys.stderr.write(cmd+"\n")
  run(shlex.split(cmd), check=True)
 # shutil.copy(args.outfile, args.outfile+".postbleu")
  cmd="{unkrep} {input} {outfile} {ttable} {unks}".format(unkrep=args.att_unk_rep, input=args.input, outfile=args.outfile, ttable=ttable, unks=unks)
  sys.stderr.write(cmd+"\n")
  run(shlex.split(cmd), check=True)
 # shutil.copy(args.outfile, args.outfile+".postunk")
  # cmd="{decfmt} {outfile} {outtmp}".format(decfmt=args.decode_format, outfile=args.outfile, outtmp=outtmp)
  # sys.stderr.write(cmd+"\n")
  # run(shlex.split(cmd), check=True)
  # shutil.copy(args.outfile, args.outfile+".postfmt")


if __name__ == '__main__':
  main()
