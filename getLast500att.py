import numpy as np
attFile = 'bigBalanceData/attentionScore.txt'
last500File = 'bigBalanceData/last500.txt'
maxLength = 1000
n = 500
output = open(last500File,'w',encoding='gbk')
for line in open(attFile):
    att = line.split(' ')
    for i in range(n,maxLength):
        output.write(str(att[i]))
        output.write(' ')
