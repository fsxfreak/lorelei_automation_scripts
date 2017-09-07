#!/usr/bin/env python
#PBS -q isi
#PBS -l walltime=336:00:00
#PBS -N watcher
# code by Leon Cheung [lcheung at isi.edu]
# Receives job numbers and model output directories, and terminates jobs
# early if one of the models earns the lowest perplexity a couple of steps in
# a row.
# Made to fit the ZophRNN framework.

import logging as logger
from collections import OrderedDict
import argparse, sys, os, time, re
import numpy as np
import subprocess

logger.basicConfig(stream=sys.stderr, level=logger.INFO,
    format='%(asctime)s [%(levelname)s]: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

# if get the worst rank for this many half epochs, stop training
TRIM_THRESHOLD = 5
LOOP_SECONDS = 120

def get_ranks(perplexities):
  # transform perplexities dict into a numpy array, in the same order
  # num_models x min(half epochs)
  perps_lens = [ len(perplexities[model]) for model in perplexities ]
  min_perps_len = min(perps_lens)
  perps = np.zeros([len(perplexities), min_perps_len])
  for i, model in enumerate(perplexities):
    perps[i] = np.array(perplexities[model][:min_perps_len])

  ranks_raw = np.transpose(np.array(
    [ np.argsort(perps[:,i]) for i in range(len(perps[0])) ]))

  ranks = OrderedDict()
  for model, rank in zip(perplexities, ranks_raw):
    ranks[model] = rank

  return ranks

def watch(job_logs):
  job_logs = job_logs[:]
  perplexities = OrderedDict()
  for job, log in job_logs:
    perplexities[job] = []

  trim_point = 0 # most recent half-epoch where we dropped a job
  while len(perplexities) > 1:
    time.sleep(LOOP_SECONDS)
   

    cont = False
    # parse out perplexities from zoph tranining log
    float_re = re.compile(r'\d+.\d+')
    for job, log in job_logs:
      try:
        with open(log, 'r') as f:
          perplexities_seen = 0
          for line in f:
            if 'Perplexity dev set' in line:
              perplexity = float(float_re.search(line).group(0))
              perplexities_seen = perplexities_seen + 1
              if perplexities_seen > len(perplexities[job]):
                perplexities[job].append(perplexity)
      except IOError: # not all models have begun training
        cont = True
        break
    if cont:
      continue

    # rank perplexities for every model so far
    ranks = get_ranks(perplexities)

    #worst_rank = len(perplexities) - 1
    #threshold = str(worst_rank) * TRIM_THRESHOLD

    # start trimming the worst model when the best model performs well
    best_rank = 0
    do_trim = False
    threshold = str(best_rank) * TRIM_THRESHOLD
    for model_index, model in enumerate(ranks):
      trim_rank = ''.join([ str(e) for e in ranks[model][trim_point:]])
      if threshold in trim_rank:
        do_trim = True
        trim_point = trim_rank.index(threshold) + trim_point + 1
        logger.debug('trim_point: %d' % trim_point)
        logger.info('model: %d is the best model so far.' % model)

    if not do_trim:
      continue

    for model_index, model in enumerate(ranks):
      # only consider ranks past the trim point
      trim_rank = ''.join([ str(e) for e in ranks[model][trim_point - 1:]])
      if int(trim_rank[0]) == len(ranks) - 1:
        # worst rank at this point in time
        del perplexities[model]
        del job_logs[model_index]

        logger.debug('trim, %s, %s' % (model, trim_point))
        logger.info('Stopping training: %d' % model)
        #subprocess.call(['qdel', '%d' % model])

        # only trim one model per iteration
        break

  # return last model num remaining
  return list(perplexities.items())[0][0]

def main():
  parser = argparse.ArgumentParser('watch jobs and early terminate them')
  parser.add_argument('--watches', type=str, required=True)
  parser.add_argument('--final_model_dir', type=str, required=True)

  args = parser.parse_args()
  log_dir = os.path.abspath(os.path.join(args.final_model_dir, 'watcher.log'))
  log_handler = logger.FileHandler(log_dir)
  log_handler.setLevel(logger.DEBUG)
  fmt = logger.Formatter('%(asctime)s [%(levelname)s]: %(message)s')
  log_handler.setFormatter(fmt)
  logger.getLogger().addHandler(log_handler)

  job_logs = [ (int(filter(str.isdigit, e.split('|||')[0])[:-1]), e.split('|||')[1]) 
                for e in args.watches.split(':') ]

  logger.info('Using a trim threshold of %d.' % TRIM_THRESHOLD)

  final_model_job = watch(job_logs)
  final_model_log = ''
  for job, log in job_logs:
    if job == final_model_job:
      final_model_log = log
  logger.debug(final_model_log)

  real_final_model_dir = os.path.abspath(os.path.join(final_model_log, '../'))
  
  final_model_dir = os.path.join(args.final_model_dir, 'model1')
  try:
    os.rmdir(final_model_dir)
  except OSError:
    pass

  try:
    os.symlink(real_final_model_dir, final_model_dir)
  except OSError:
    logger.warning('a trained model already exists at %s.' % final_model_dir)

  logger.info('Final model taken from %s to %s.'
      % (real_final_model_dir, final_model_dir))

if __name__ == '__main__':
  logger.getLogger().setLevel(logger.DEBUG)
  main()
