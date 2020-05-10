import numpy as np
xtrain = []
ytrain = []
ids = []
label0seq = []
label1seq = []
seqsLen = []
p = 0.0005
n = 500
train_file = 'bigBalanceData/trainfile.csv'
attFile = 'bigBalanceData/last500.txt'
seqs_label1_file = 'bigBalanceData/p=0.0005/seqs_label1.fasta'
seqs_label0_file = 'bigBalanceData/p=0.0005/seqs_label0.fasta'
seqs_len_file = 'bigBalanceData/p=0.0005/seqs_len.txt'

for line in open(train_file):
    id,seq,label = line.split(',')
    ids.append(id)
    xtrain.append(seq)
    ytrain.append(label)
seqs_num = len(ytrain)
attention_layer_output = np.loadtxt(attFile)
print("len attention_layer_output",len(attention_layer_output))

# get segments from indexs
def getSeq(labelseq,sequence,indexs):
    index = indexs[0] + n
    length = len(indexs)
    if(index > len(sequence)):
        print('index：',index)
        print('indexs[0]',indexs[0])
        print('n',n)
        print('len(sequence):',len(sequence))
        print('sequence',sequence)
        exit()
    if(sequence[index] == '0'):
        return
    if (length != 0):
        labelseq.append(sequence[index:index + length])

# Find indexs with continuous length greater than 8
def find_(a):
    flag = 0
    lens = 0
    indexList = []
    for i in range(0,len(a) -1):
        if (a[i] == a[i + 1] - 1) :
            lens = lens + 1
            flag = i
        else:
            if(lens >= 8):
                temp = [x for x in a[flag + 1 - lens:flag + 1 + 1]]
                if(len(temp) == 0):
                    print('temp',temp)
                indexList.append(temp)
            lens = 0
    return indexList


for j in range(seqs_num):
    id = ids[j]
    label = ytrain[j]
    label = label.strip('\n')
    sequence = xtrain[j]
    indexList= find_([i for i, v in enumerate(attention_layer_output[j])if float(v) > p])
    if(label == '0'):
        for i in range(len(indexList)):
            getSeq(label0seq,sequence,indexList[i])
    if(label == '1'):
        for i in range(len(indexList)):
            getSeq(label1seq,sequence,indexList[i])
for i in range(len(label1seq)):
    seqsLen.append(len(label1seq[i]))
seqs_label0 = open(seqs_label0_file, 'w', encoding='gbk')
seqs_label1 = open(seqs_label1_file, 'w', encoding='gbk')
seqsLen_output = open(seqs_len_file, 'w', encoding='gbk')

for i in range(len(seqsLen)):
    seqsLen_output.write(str(seqsLen[i]) + '\n')
for i in range(len(label1seq)):
    seqs_label1.write('>'+str(i+1)+'\n')
    seqs_label1.write(label1seq[i])
    seqs_label1.write('\n')
for i in range(len(label0seq)):
    seqs_label0.write('>' + str(i+1) + '\n')
    seqs_label0.write(label0seq[i])  
    seqs_label0.write('\n')  


