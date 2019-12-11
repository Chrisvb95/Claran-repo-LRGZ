import numpy as np 
import matplotlib.pyplot as plt

recs = []
precs = []


for i in range(1,4):
    recs.append(np.genfromtxt('rec'+str(i)+'.csv'))
    precs.append(np.genfromtxt('prec'+str(i)+'.csv'))
    print(len(recs[i-1]))
    print(len(precs[i-1]))

print(recs[2])
print(precs[2])


marker = '.'

plt.scatter(recs[0],precs[0],marker=marker,label='1')
plt.plot(recs[0],precs[0])
plt.scatter(recs[1],precs[1],marker=marker,label='2')
plt.plot(recs[1],precs[1])
plt.scatter(recs[2],precs[2],marker=marker,label='3')
plt.plot(recs[2],precs[2])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.title('Precision vs Recall')
plt.savefig('visualisation/rec_prec.png')


'''
~~~~~~~~
True positives: 223.0
False positives: 223.0
AP for 1 = 0.3187
~~~~~~~~
True positives: 15.0
False positives: 35.0
AP for 2 = 0.1740
~~~~~~~~
True positives: 6.0
False positives: 23.0
AP for 3 = 0.3502
Mean AP = 0.2809
~~~~~~~~
Results:
0.319
0.174
0.350
0.281
~~~~~~~~
'''