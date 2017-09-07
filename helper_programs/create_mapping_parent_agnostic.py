#!/usr/bin/env python
# -*- coding: utf-8 -*-
import codecs
import sys
from itertools import izip
from collections import defaultdict as dd

if len(sys.argv)!=6:
  print "USAGE: <source file> <target file> <count threshold> <output model file> <big source file>"
  sys.exit()
source_file = codecs.open(sys.argv[1],'r','utf-8')
target_file = codecs.open(sys.argv[2],'r','utf-8')
count_threshold = int(sys.argv[3])
output_model_file = codecs.open(sys.argv[4],'w','utf-8')
source_file_big = codecs.open(sys.argv[5],'r','utf-8')

source_counts = dd(int)
source_big_counts = dd(int)
target_counts = dd(int)
source_words = set([])
source_words_big = set([])

for line_s in source_file:
  line_s = line_s.replace('\n','').split(' ')
  for word in line_s:
    source_counts[word]+=1
for line_t in target_file:
  toks = line_t.split('\t')
  target_counts[toks[0]] = int(toks[1])
print >> sys.stderr, len(target_counts)

for line in source_file_big:
  line = line.replace('\n','').split(' ')
  for word in line:
    source_big_counts[word]+=1

for tup in source_counts:
  if source_counts[tup] >= count_threshold:
    source_words.add(tup)

target_counts_sorted = sorted(target_counts.items(), key=lambda x: x[1], reverse=True)
target_words = []
i = 0
for tup in target_counts_sorted:
  if i >= 40000:
    break
  if tup[1] >= count_threshold and tup[0] in source_big_counts:
    i += 1
    target_words.append(tup[0])

print >> sys.stderr, "Number of unique source words above count threshold:",len(source_words)
print >> sys.stderr, "Number of unique target words above count threshold:",len(target_words)

import operator
sorted_big_counts = sorted(source_big_counts.items(), key=operator.itemgetter(1))[::-1][:len(source_words)]

for tup in sorted_big_counts:
  source_words_big.add(tup[0])

index = 1
output_model_file.write('1 1 '+ str(len(target_words)+3) + ' ' + str(len(source_words)+1) +'\n')
output_model_file.write('==========================================================\n')
output_model_file.write('0 <UNK>\n')
for word in source_words_big:
  output_model_file.write(str(index) + ' ' + word + '\n')
  index+=1

index = 3
output_model_file.write('==========================================================\n')
output_model_file.write('0 <START>\n')
output_model_file.write('1 <EOF>\n')
output_model_file.write('2 <UNK>\n')
for word in target_words:
  output_model_file.write(str(index) + ' ' + word + '\n')
  index+=1
output_model_file.write('==========================================================\n')

