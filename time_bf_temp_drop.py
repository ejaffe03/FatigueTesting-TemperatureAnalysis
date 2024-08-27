# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 09:48:31 2024

@author: ellie
"""


#Calculate 
#%% imports
import matplotlib.pyplot as plt #for created plots
import matplotlib as mpl
mpl.rc("figure", dpi=600)
import csv # this gives us tools to manipulate CSV files
#different mathmatical functions:
from datetime import datetime
import pandas as pd
import datetime as dtm
import math
import matplotlib.dates as mdates
#%% setting up blank dictionary and opening files
newlist1=[]
def nearest(items, pivot):
    # newitems=[i for i in items if i<=pivot]
    # print(newitems)
    # newlist1.extend(newitems)
    # if not newitems:
        # return min(items, key=lambda x: abs(x - pivot))
    # if newitems:
    minimum=min(items, key=lambda x: abs(x - pivot))
    if (pivot-minimum)<dtm.timedelta(days=1):
        return minimum
        # else:
            # return min(items, key=lambda x: abs(x - pivot)) 
behavior={}
temp={}
file1 = open(r"C:\\Users\ellie\Downloads\UROP\Temperature Analysis\parsed-data.csv")
csvreader = csv.reader(file1)
next(csvreader)
next(csvreader)
j=0
k=0
m=0
#%%reading file and creating dictionaries for temperature and behavior events
#go thru row, record time, check in drinking range, subtract difference if positive/0, check if in dictionary, if replacement is smaller, replace
for row in csvreader:
    temptime=datetime.strptime(row[2]+" "+row[3], '%m/%d/%Y %I:%M:%S %p')
    starttime=datetime.strptime(row[2]+" "+row[6], '%m/%d/%Y %I:%M:%S %p')
    endtime=datetime.strptime(row[2]+" "+row[7], '%m/%d/%Y %I:%M:%S %p')

    if temptime>=starttime and temptime<=endtime:
        timediff=dtm.timedelta(days=0)
    elif temptime<=starttime:
        timediff=temptime-starttime
    else:
        timediff=temptime-endtime
        
    if temptime not in list(temp.keys())and timediff>=dtm.timedelta(days=0):
        temp[temptime]=timediff
    elif timediff>=dtm.timedelta(days=0) and timediff<temp[temptime]:
        temp[temptime]=timediff
file1.close()
newlist=[1]*len(temp.values())
df = pd.DataFrame(list(zip(list(temp.values()), newlist)),columns =['DeltaT', 'val'])
df.index=df['DeltaT']
df=df.drop(columns=['DeltaT'])
df=df.sort_index()
df2=df
df=df.resample('1min').sum()
df.at[df.index[-1], 'val'] = 1
df1=df['val']

#%% this adds behavior information if uncommented
total=0

values=list(df['val'])
index=list(df.index)
newlist=[]
for x in values:
    total+=x
    newlist.append(total/175)
df=pd.DataFrame(list(zip(index, newlist)),columns =['DeltaT', 'val'])
df.index=df['DeltaT']
df=df.drop(columns=['DeltaT'])
#%%plotting bar graph
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = "9"

# fig = plt.figure(figsize=(10, 4))
plt.title("Probability of Event before Temp Drop")
plt.plot(df.index.astype('timedelta64[s]')/60,df['val'], color='k', linewidth=.7)
plt.xlim(-3,30)
# plt.ylim(0,1)
plt.gca().invert_xaxis()
# plt.axvline(0, linewidth=.7)
# plt.axhline(0, linewidth=.7)


plt.xlabel('Minutes Before Temperature Drop')
plt.ylabel('Probability of Ingestion')
# plt.title('Probability of Temperature Drop caused by Behavior event over the course of a day')

plt.show()


