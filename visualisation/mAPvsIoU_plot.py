import pickle
import numpy as np
import matplotlib.pyplot as plt

aps = np.genfromtxt('mAPvsIoU.csv',dtype=float,delimiter=',')
ovthresh = np.arange(0,1,0.01)
marker = '.'

plt.grid(b=True, which='major', color='#666666', linestyle='-',alpha=0.5)
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.axvline(0.5,color='k',linestyle='--')
plt.scatter(ovthresh,aps[:,0],marker=marker,label='1')
plt.plot(ovthresh,aps[:,0])
plt.scatter(ovthresh,aps[:,1],marker=marker,label='2')
plt.plot(ovthresh,aps[:,1])
plt.scatter(ovthresh,aps[:,2],marker=marker,label='3')
plt.plot(ovthresh,aps[:,2])
plt.scatter(ovthresh,aps[:,3],marker=marker,label='Mean')
plt.plot(ovthresh,aps[:,3])
plt.xlabel('IoU threshold')
plt.ylabel('mAP')
plt.xlim(left=0,right=1)
plt.ylim(bottom=0)
plt.legend()
plt.title('Mean average precision as a function of IoU')
plt.savefig('visualisation/mAPvsIoU.png')

