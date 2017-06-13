#! /usr/bin/env python3
import argparse
import sys
#from sbmt import parse_rule, parse_nbest
import re

def parse_feat_string(string):
  '''
  given an isi-style string of space separated key=val pairs and key={{{entry with spaces}}} pairs, return
  a dict of those entries. meant to be used by various flavors, i.e. nbest list, rule
  '''
  feats={}
  spos=0
  entryre=re.compile(r"\s*([^\s=]+)=((?:[^\s{}]+)|(?:{{{.*?}}}))\s*")
  for match in entryre.findall(string):
    #print("%s -> %s" % (match[0], match[1]))
    feats[match[0]]=match[1]
  return feats

def parse_nbest(string):
  '''
  given an isi-style nbest entry headed with NBEST, return the feature dictionary
  '''
  fields=string.split()
  if fields[0] != "NBEST":
    raise Exception("String should start with NBEST but starts with "+fields[0])
  return parse_feat_string(' '.join(fields[1:]))

def parse_rule(string):
  '''
  given an isi syntax rule, return the feature dictionary including "SOURCE" and "TARGET"
  '''
  try:
    (rule, rest) = string.split(" ### ")
    feats = parse_feat_string(rest)
    (target, source) = rule.split(" -> ")
    feats['SOURCE'] = source
    feats['TARGET'] = target
    return feats
  except Exception as e:
    raise Exception("could not parse "+string, e)



def main():
  parser = argparse.ArgumentParser(description="Get a statistic from a file of n-best or rules ")
  parser.add_argument("--infile", "-i", nargs='?', type=argparse.FileType('r'), default=sys.stdin, help="input file")
  parser.add_argument("--outfile", "-o", nargs='?', type=argparse.FileType('w'), default=sys.stdout, help="output file")
  parser.add_argument("--rules", "-r", action='store_true', default=False, help="expect sbmt rules (otherwise expect nbest)")
  parser.add_argument("--statistics", "-s", nargs='+', help="list of statistics to extract")
  parser.add_argument("--label", "-l", default=False, action='store_true', help="print feature label with feature")
  parser.add_argument("--inverse", "-v", default=False, action='store_true', help="statistics list is set of exclusions, not inclusions")


  try:
    args = parser.parse_args()
  except IOError as msg:
    parser.error(str(msg))

  parse_fn = parse_rule if args.rules else parse_nbest
  for line in args.infile:
      entry=parse_fn(line.strip())
      vals = []
      if args.inverse:
        statset = sorted(set(entry.keys())-set(args.statistics))
      else:
        statset = [x for x in args.statistics if x in entry]
      for stat in statset:
        if args.label:
          vals.append(stat)
        vals.append(entry[stat])
      args.outfile.write('\t'.join(vals)+"\n")

if __name__ == '__main__':
  main()
