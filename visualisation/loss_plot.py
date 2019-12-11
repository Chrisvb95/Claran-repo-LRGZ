import numpy as np 
from numpy import genfromtxt
from matplotlib import pyplot as plt 
from matplotlib import patches
import csv 
import io
import sys
import os 
import PIL.Image as im
sys.path.insert(0,os.getcwd()+'/visualisation/')
from CLARAN_cutout_object import cutout_CLARAN_labeled

path_csv = 'Results/loss/'
path_results = 'data/RGZdevkit2017/results/LRGZ2017/Main/'
path_cutouts = 'data/RGZdevkit2017/RGZ2017/PNGImages/'
'''
# Importing and plotting the loss funtion 
loss_track_file = 'loss_track.csv'
with open(loss_track_file,'r') as myfile:
    data = myfile.read().replace('"','')
loss = genfromtxt(io.StringIO(data),delimiter=',')
plt.plot(loss)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title(f'Loss function - {len(loss)} Iterations and  training samples')
plt.savefig('visualisation/loss_plot.png')
plt.close()
'''

def csv_read(name):
    with open(name,'r') as myfile:
        data = myfile.read().replace('"','')
        return genfromtxt(io.StringIO(data),delimiter=',')

loss1 = csv_read(path_csv+'loss_track_50000it_3samples_3labels.csv')[:5000]
loss2 = csv_read(path_csv+'loss_track_5000it_215samples_withboxes_out.csv')
loss3 = csv_read(path_csv+'loss_track_5000it_215samples_withboxes_edge.csv')
loss4 = csv_read(path_csv+'loss_track_5000it_160samples_clean.csv')


plt.plot(loss2, label='215 samples exceed edges',color='C3')
plt.plot(loss3, label='215 samples touch edges',color='C2')
plt.plot(loss4, label='160 \'clean\' samples',color='C0')
plt.plot(loss1, label='3 samples',color='C1')

plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title(f'Loss function against iterations including \n bounding boxes outside of bounds')
plt.legend()
plt.savefig('visualisation/loss_plot_5000_edges.png')
plt.close()


plt.plot(loss4, label='160 \'clean\' samples')
plt.plot(loss1, label='3 samples')

plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title(f'Loss function against iterations')
plt.legend()
plt.savefig('visualisation/loss_plot_5000_clean.png')
plt.close()


loss5 = csv_read(path_csv+'loss_track_50000it_396flipsamples_new.csv')
loss6 = csv_read(path_csv+'loss_track_50000it_800flipsamples_new.csv')
#loss7 = csv_read(path_csv+'loss_track_50000it_800flipssamples_3labels.csv')
loss8 = csv_read(path_csv+'loss_track_100000_800_flipsamples_3labels.csv')


#plt.plot(loss7, label='215 samples touch edges',color='C2')
plt.plot(loss6, label='789 samples 6 classes',color='C0')
plt.plot(loss5, label='396 samples 6 classes',color='C1')

plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title(f'Loss function against iterations for different samples sizes')
plt.legend()
plt.savefig('visualisation/loss_plot_50000.png')
plt.close()

plt.plot(loss8, label='789 samples 3 classes')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title(f'Loss function against iterations for final training sample')
plt.legend()
plt.savefig('visualisation/loss_plot_100000.png')
plt.close()