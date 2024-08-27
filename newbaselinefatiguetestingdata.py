# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 11:01:17 2024

@author: ellie
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 14:17:43 2024

@author: ellie
"""

#!/usr/bin/env python
# coding: utf-8

#%% imports

import numpy as np # works with complex but one of the arguments must be complex
import matplotlib.pyplot as plt #for created plots
import csv # this gives us tools to manipulate CSV files
#different mathmatical functions:
from scipy.signal import find_peaks
import math
import os
import pandas as pd
import matplotlib as mpl
# Nedded for ALS
from scipy import sparse
from scipy.linalg import cholesky
from scipy.sparse.linalg import spsolve
mpl.rc("figure", dpi=1200)


def als(y, lam=1e6, p=0.1, niter=10):
    r"""
    Implements an Asymmetric Least Squares Smoothing
    baseline correction algorithm (P. Eilers, H. Boelens 2005)
 
    Baseline Correction with Asymmetric Least Squares Smoothing
    based on https://web.archive.org/web/20200914144852/https://github.com/vicngtor/BaySpecPlots
 
    Baseline Correction with Asymmetric Least Squares Smoothing
    Paul H. C. Eilers and Hans F.M. Boelens
    October 21, 2005
 
    Description from the original documentation:
 
    Most baseline problems in instrumental methods are characterized by a smooth
    baseline and a superimposed signal that carries the analytical information: a series
    of peaks that are either all positive or all negative. We combine a smoother
    with asymmetric weighting of deviations from the (smooth) trend get an effective
    baseline estimator. It is easy to use, fast and keeps the analytical peak signal intact.
    No prior information about peak shapes or baseline (polynomial) is needed
    by the method. The performance is illustrated by simulation and applications to
    real data.
 
 
    Inputs:
        y:
            input data (i.e. chromatogram of spectrum)
        lam:
            parameter that can be adjusted by user. The larger lambda is,
            the smoother the resulting background, z
        p:
            wheighting deviations. 0.5 = symmetric, <0.5: negative
            deviations are stronger suppressed
        itermax:
            number of iterations to perform
    Output:
        the fitted background vector
 
    """
    L = len(y)
    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
    D = lam * D.dot(D.transpose()) # Precompute this term since it does not depend on `w`
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)
    for i in range(niter):
        W.setdiag(w) # Do not create a new matrix, just update diagonal values
        Z = W + D
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    for x in range(0,L):
        y[x]=y[x]-z[x]
    return y

#%%data parsing functions

## TRT iRES v2 Rev1


#combining the data in one day while skipping the titles and labels in the files and baseline correcting and finding the peaks
def filereading(directory,files):
    #data parsing
    data=[]
    deltat=0
    for x in files:
        x=open(os.path.join(directory, x),'r')
        csvreader = csv.reader(x)
        next(csvreader)
        next(csvreader)
        next(csvreader)
        next(csvreader)
        for row in csvreader:
            #time automatically starts at 0 for each test- this increases the time so that it counts
            #total time the device is run
            row[0]=float(row[0])+float(deltat)
            data.append(row)
        x.close()
        newnumber=data[-1]
        deltat=newnumber[0]
    #data starts at index 4
    print(deltat)
    subdata=data[0:len(data)]
    n=len(subdata)
    #putting data into 2 np arrays- time and magnitude
    t_TRT_SGF=np.empty(n)
    load_TRT_SGF=np.empty(n)
    i=0
    while i<n:
        t_TRT_SGF[i]=subdata[i][0] # times at which each of the samples were taken
        load_TRT_SGF[i]=subdata[i][1] # first channel of Digilent Discovery 2
        i=i+1
    #baseline correcting
    print(i)
    # plt.plot(t_TRT_SGF, load_TRT_SGF)
    df=pd.DataFrame(list(zip(list(t_TRT_SGF),list(load_TRT_SGF))))
    baseline_als_raman = als(load_TRT_SGF, lam=1e6)
    maxpeaks, _=find_peaks(baseline_als_raman, height=(None,-1))
    finalized=[]
    for x in maxpeaks:
        a=list(range(x-50,x+50))
        if not any(y in a for y in finalized):
            finalized.append(list(range(x-50,x+50)))
    print(maxpeaks)
    baseline_als_raman=np.delete(baseline_als_raman, finalized, axis=0)
    t_TRT_SGF=np.delete(t_TRT_SGF, finalized, axis=0)
    # plt.scatter(t_TRT_SGF[maxpeaks],baseline_als_raman[maxpeaks], color='k')
    # plt.show()
    # plt.plot(t_TRT_SGF,baseline_als_raman)
    #finding peaks    

    print(1)
    peak2peak, _=find_peaks(baseline_als_raman,distance=55)
    print(1)
    #setting x axis as number of cycles
    new_t_TRT=[i for i in range(0,len(peak2peak))]
    #finding values of magnitude using the peak2peak indices
    newpeaks=[baseline_als_raman[x] for x in peak2peak]
    #turn into dataframe
    dictionary=dict(zip(new_t_TRT,newpeaks))
    df = pd.DataFrame.from_dict(dictionary, orient='index', columns=['Force'])
    return df

#organizing files by test
def readingfiles(directory):
    for root, dirs, files in os.walk(directory):
        print(1)
        df=filereading(directory,files)
    return df

#%%Opening directories and sending data to be parsed
directory = (r"C:\Users\ellie\Downloads\July-15-24-0mm")
directory1 = (r"C:\Users\ellie\Downloads\July-17-24-2mmx2")
directory2=(r"C:\Users\ellie\Downloads\July 22-24-2mm x1")
directory3=(r"C:\Users\ellie\Downloads\July-26-24 2mmx2")
directory4=(r"C:\Users\ellie\Downloads\July 29-24 1mmx1")
directory5=(r"C:\Users\ellie\Downloads\July-31-24 1mmx1")
directory6=(r"C:\Users\ellie\Downloads\August-2-24 0mm")
directory7=(r"C:\Users\ellie\Downloads\August-5-24 2mmx1")
directory8=(r"C:\Users\ellie\Downloads\August-8-24 1mm")
directory9=(r"C:\Users\ellie\Downloads\August-12-24 0mm")
directory10=(r"C:\Users\ellie\Downloads\August-12-24 2mmx1")
directory11=(r"C:\Users\ellie\Downloads\August-14-24 2mmx2")


df=readingfiles(directory)
df1=readingfiles(directory1)
df2=readingfiles(directory2)
df3=readingfiles(directory3)
df4=readingfiles(directory4)
df5=readingfiles(directory5)
df6=readingfiles(directory6)
df7=readingfiles(directory7)
df8=readingfiles(directory8)
df9=readingfiles(directory9)
df10=readingfiles(directory10)
df11=readingfiles(directory11)

# plt.plot(df2.index,df2['Force'])
# plt.plot(df7.index,df7['Force'])

# df2=df2.rolling(100, center=True).sum()/100
# df1=df1.rolling(100, center=True).sum()/100
# df2=df2.rolling(100, center=True).sum()/100
# df4=df4.rolling(100, center=True).sum()/100

df13=pd.concat([df1,df3, df11], axis=1)
df13=df13.ffill()
df13['ave']=df13.sum(axis=1, numeric_only=True)/3
df13=df13.rolling(100, center=True).sum()/100
df13['std']=df13['Force'].std(axis=1)/math.sqrt(3)

df14=pd.concat([df4,df5,df8], axis=1)
df14=df14.ffill()
df14['ave']=df14.sum(axis=1, numeric_only=True)/3
df14=df14.rolling(100, center=True).sum()/100
df14['std']=df14['Force'].std(axis=1)/math.sqrt(3)

df15=pd.concat([df,df6, df9], axis=1)
df15=df15.ffill()
df15['ave']=df15.sum(axis=1, numeric_only=True)/3
df15=df15.rolling(100, center=True).sum()/100
df15['std']=df15['Force'].std(axis=1)/math.sqrt(3)

df16=pd.concat([df2,df7,df10], axis=1)
df16=df16.ffill()
df16['ave']=df16.sum(axis=1, numeric_only=True)/3
df16=df16.rolling(100, center=True).sum()/100
df16['std']=df16['Force'].std(axis=1)/math.sqrt(3)

#%%plotting


#creating and labelling a plot
# # plt.fill_between(df2.index, df2['ave']-df2['std'], df2['ave']+df2['std'], alpha=.2)

# plt.plot(df2.index,df2['Force'], label='passivated with SGF2')

# plt.plot(df9.index,df9['Force'])
# plt.plot(df.index,df['Force'])
# plt.plot(df6.index,df6['Force'])

# plt.plot(df16.index, df16['ave'], label='2mm Radius 1 arm N=2')
# plt.plot(df13.index, df13['ave'], label='2mm Radius 2 arms N=2')
# plt.plot(df14.index, df14['ave'], label='1mm Radius 1 arm N=3')
# # plt.plot(df4.index,df4['Force'],label='2mm bend radius with 2 arms 2')
# # plt.plot(new_a2_x, new_a2_y, label='Passivated with SGF3')
# plt.plot(df15.index,df15['ave'], label='0mm Radius 1 arm N=3')

# plt.xlabel('Cycles')
# plt.ylabel('Load [N]')
# plt.title('Applied Force over Repeated Cycles')
# plt.legend(loc="upper right")
# plt.show()


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = "7"
fig = plt.figure(figsize=(9,3))

gs = fig.add_gridspec(1,4, wspace=.1)
axs = gs.subplots(sharex=True, sharey=True)

axs[0].set_box_aspect(aspect=1)
axs[1].set_box_aspect(aspect=1)
axs[2].set_box_aspect(aspect=1)
axs[3].set_box_aspect(aspect=1)

axs[2].plot(df16.index, df16['ave'], label="2mm Radius 1 arm N=2", color='k',linewidth=.5)
axs[2].fill_between(df16.index, df16['ave']-df16['std'], df16['ave']+df16['std'], alpha=.2)

axs[3].plot(df13.index, df13['ave'], label="2mm Radius 2 arms N=2", color='k',linewidth=.5)
axs[3].fill_between(df13.index, df13['ave']-df13['std'], df13['ave']+df13['std'], alpha=.2)

axs[1].plot(df14.index, df14['ave'], label="1mm Radius 1 arms N=3", color='k',linewidth=.5)
axs[1].fill_between(df14.index, df14['ave']-df14['std'], df14['ave']+df14['std'], alpha=.2)

axs[0].plot(df15.index, df15['ave'], label="0mm Radius 1 arms N=3", color='k',linewidth=.5)
axs[0].fill_between(df15.index, df15['ave']-df15['std'], df15['ave']+df15['std'], alpha=.2)

ax=axs.flat
ax[0].set(ylabel='Load')
ax[0].set_title(label='0mm Radius 1 arms N=3')
ax[1].set_title(label='1mm Radius 1 arms N=3')
ax[2].set_title(label='2mm Radius 1 arms N=3')
ax[3].set_title(label='2mm Radius 2 arms N=3')

ax[0].set(xlabel='Cycles')
ax[1].set(xlabel='Cycles')
ax[2].set(xlabel='Cycles')
ax[3].set(xlabel='Cycles')



plt.show()
