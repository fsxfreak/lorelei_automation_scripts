import codecs
import sys
import re

input_file_name = str(sys.argv[1])
input_file = codecs.open(input_file_name,'r','utf-8')

outputs= []
for line in input_file:
	re.sub('\n','',line)
	line = line.split(' ')
	if line[0]=="<START>":
		del line[0]
		del line[-1]
		outputs.append(' '.join(line)+'\n')

input_file = codecs.open(input_file_name,'w','utf-8')	
for line in outputs:
	input_file.write(line)

