import codecs
import sys

data_file = codecs.open(sys.argv[1],'r','utf-8')
output_file = codecs.open(sys.argv[1] + '.tab' ,'w','utf-8')

counter = 1
for line in data_file:
	output_file.write(str(counter) + '\t' + line)
	counter+=1
