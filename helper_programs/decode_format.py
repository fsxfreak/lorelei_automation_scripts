import codecs
import sys
import re

input_file_name = str(sys.argv[1])
input_file = codecs.open(input_file_name,'r','utf-8')
orig_file = codecs.open(sys.argv[2],'r','utf-8')
#output_file = codecs.open(output_file_name,'w','utf-8')

unked_trans = [line.replace('\n','') for line in input_file]
trans = []
curr_src_sent_num = -1
curr_decode_num = 0
prev_score = '0'
for line in orig_file:
	line_orig = line
	line = line.replace('\n','').split(' ')
	if line[0]=="<START>":
		del line[0]
		del line[-1]
		trans.append(str(curr_src_sent_num)+'\t'+str(curr_decode_num)+'\t'+prev_score+'\t'+unked_trans[curr_src_sent_num]+'\n')
		#output_file.write(str(curr_src_sent_num)+'\t'+str(curr_decode_num)+'\t'+prev_score+'\t'+' '.join(line)+'\n')
		curr_decode_num+=1
	elif list(line_orig)[0] == "-" and list(line_orig)[1] == '-':
		curr_src_sent_num+=1
		curr_decode_num = 0
	elif line[0] == '-Score:':
		prev_score = line[1]

output_file = codecs.open(input_file_name,'w','utf-8')
for line in trans:
	output_file.write(line)

