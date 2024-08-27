# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 11:41:27 2024

@author: ellie
"""
#Calculate True Positives and Time of Day
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
import numpy as np
from matplotlib.ticker import FuncFormatter

#%% setting up blank dictionary and opening files

behavior={}
temp={}
file1 = open(r"C:\Users\ellie\Downloads\UROP\Temperature Analysis\parsed-data.csv")
csvreader = csv.reader(file1)
next(csvreader)
next(csvreader)
j=0
k=0
m=0
#%%reading file and creating dictionaries for temperature and behavior events
for row in csvreader:
    if row[5]=='drinking':
        behavior[f'd{j}']=datetime.strptime(row[2]+" "+row[6], '%m/%d/%Y %I:%M:%S %p')
        j+=1
        behavior[f'd{j}']=datetime.strptime(row[2]+" "+row[7], '%m/%d/%Y %I:%M:%S %p')
        j+=1
    elif 'eating:' in row[5]:
        behavior[f'f{k}']=datetime.strptime(row[2]+" "+row[6], '%m/%d/%Y %I:%M:%S %p')
        k+=1
        behavior[f'f{k}']=datetime.strptime(row[2]+" "+row[7], '%m/%d/%Y %I:%M:%S %p')
        k+=1
    if datetime.strptime(row[2]+" "+row[3], '%m/%d/%Y %I:%M:%S %p') not in temp:
        temp[f't{m}']=datetime.strptime(row[2]+" "+row[3], '%m/%d/%Y %I:%M:%S %p')
        m+=1
        temp[f't{m}']=datetime.strptime(row[2]+" "+row[4], '%m/%d/%Y %I:%M:%S %p')
        m+=1
file1.close()

#%% this function creates a dataframe with all times, and a 1 if there was an event ongoing, and 0 if there was no event ongoing, and then divided into 15 min segments 
#in the day with probability, and a std column. it strips it of the date and then returns it
def behavior_probability(behavior1):
    l=len(behavior1.values())-2
    # print(behavior1.keys())
    behavior2={}
    n=0
    while n<l:
        start_date=list(behavior1.values())[n]
        n+=1
        end_date=list(behavior1.values())[n]
        delta=dtm.timedelta(seconds=1)
        while (start_date<=end_date):
            behavior2[start_date]=1
            start_date+=delta
        n+=1
        second_end_date=list(behavior1.values())[n]
        while (end_date<=second_end_date):
            behavior2[end_date]=0
            end_date+=delta
    df = pd.DataFrame.from_dict(behavior2, orient='index', columns=['DateValue'])
    df.index.name = 'Date'
    # print(len(df.index))
    std=df.resample('30min').std()/math.sqrt(36)
    by10=df.resample('30min').sum()/(1800)
    by10['std']=std
    by10.index=by10.index.time
    return by10
#%% this adds behavior information if uncommented
# newbehavior= behavior_probability(behavior)

date = str(datetime.strptime('2018-01-01', '%Y-%m-%d').date())
# newbehavior=newbehavior.sort_index()


# newbehavior.index = pd.to_datetime(date + " " + newbehavior.index.astype(str))
# newbehavior.index.name = 'Date'
# newbehavior=newbehavior.resample('10min').sum()/36
# df_norm1 = (newbehavior-newbehavior.min())/(newbehavior.max()-newbehavior.min())


#%%This runs the function above for the temperature information, then makes sure that the days are all combined into 1 day, and adds a random date so that it can be graphed
newtemp=behavior_probability(temp)
newtemp.index.name = 'Date'
newtemp=newtemp.sort_index()
newtemp.index = pd.to_datetime(date + " " + newtemp.index.astype(str))
newtemp=newtemp.resample('30min').sum()/36
index=newtemp.index.to_list()

#%%normalizing, and mask allows all the 0% to be removed if the second line is uncommented
df_norm = (newtemp-newtemp.min())/(newtemp.max()-newtemp.min())
mask=newtemp['DateValue']==0
# newtemp=newtemp[~mask]

def my_formatter(x, pos):
    """Format 1 as 1, 0 as 0, and all values whose absolute values is between
    0 and 1 without the leading "0." (e.g., 0.7 is formatted as .7 and -0.4 is
    formatted as -.4)."""
    val_str = '{:g}'.format(x)
    if np.abs(x) > 0 and np.abs(x) < 1:
        return val_str.replace("0", "", 1)
    else:
        return val_str

#%%plotting bar graph
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = "8"

fig = plt.figure(figsize=(5, 2))

plt.xlabel('Time of Day')
plt.ylabel('Probability of Ingestion')
# plt.title('Probability of Temperature Drop caused by Behavior event over the course of a day')
width=dtm.timedelta(minutes=20)
plt.bar(newtemp.index, newtemp['DateValue'], label="Probability of Temperature Drop", width=width)
plt.errorbar(newtemp.index, newtemp['DateValue'], yerr=newtemp['std'], 
              label="Probability of temp Event", fmt='o',marker='',color='k', capsize=1, elinewidth=.5)
plt.axvspan(pd.to_datetime('2018-01-01 0:00:00'), pd.to_datetime('2018-01-01 7:00:00'), color='grey', alpha=0.3, lw=0)
plt.axvspan(pd.to_datetime('2018-01-01 19:00:00'), pd.to_datetime('2018-01-01 23:59:59'), color='grey', alpha=0.3, lw=0)
myFmt = mdates.DateFormatter('%H:%M')
plt.gca().xaxis.set_major_formatter(myFmt)
major_formatter = FuncFormatter(my_formatter)
plt.gca().yaxis.set_major_formatter(major_formatter)

plt.show()


