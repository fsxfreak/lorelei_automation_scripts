import codecs
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re

input_data = codecs.open(sys.argv[1],'r','utf-8')
output_dir = sys.argv[2]
num = sys.argv[3]
info_dump = codecs.open(sys.argv[4],'a','utf-8')

#Now parse the output file
data = [] # (epoch num, perp)

curr_epoch = 0.5
for line in input_data:
	if re.search("New dev set Perplexity",line):
		perp = line.replace('\n','').split(' ')[-1]
		data.append((curr_epoch,perp))
		curr_epoch+=0.5

#print(data)
plt.plot([tup[0] for tup in data],[tup[1] for tup in data])
plt.xlabel('Epoch')
plt.ylabel('Dev Perplexity')
plt.title('Model '+num)
plt.grid()
plt.savefig(output_dir+'model'+num+'.png')

#now output the data to the info file
info_dump.write('Info for Model '+num+'\n')
info_dump.write('---------------------------------------------------\n')
info_dump.write('Epoch\tDev Perp\n')
for x,y in data:
	info_dump.write(str(x)+'\t'+str(y)+'\n')

info_dump.write('\n\n\n')
