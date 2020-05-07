import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from time import time

def get_hists(a,b,pos,neg,t0):
    ind = np.arange(b - a + 2)
    pos_hist = np.histogram(pos, bins=ind)
    ind = ind[:-1]
    p0=pos_hist[0]
    p1=pos_hist[1]

    ind= np.array([p1[i] for i in range(len(p0)) if p0[i]!=0])
    p0 = np.array([p0[i] for i in range(len(p0)) if p0[i]!=0])

    plt.figure()
    ppos = plt.bar(ind, p0,width=70.0)
    plt.title('Frauds as a function of time (in seconds)')
    # plt.legend(ppos[0]), 'pos')
    print('total runtime is {} seconds'.format(time()-t0))
    # plt.show()

def get_hists_grouped(a,b,step,pos,t0,bar_width):
    ind=np.arange(a,b,step)
    pos_hist = np.histogram(pos,bins=ind)
    ind=(ind[:-1])//step
    p0=pos_hist[0]

    plt.figure()
    ppos = plt.bar(ind, p0,width=bar_width,alpha=0.5)
    # sns.kdeplot(p0)

    plt.title('Frauds as a function of time (every {} seconds)'.format(step))
    print('total runtime is {} seconds'.format(time()-t0))
    plt.xlabel('time ({} seconds)'.format(step))
    plt.ylabel('frauds')
