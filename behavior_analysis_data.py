# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 12:48:59 2024

@author: ellie
"""
#Calculate False Negatives and confusion matrix
import matplotlib.pyplot as plt #for created plots
import matplotlib as mpl
mpl.rc("figure", dpi=200)
import csv # this gives us tools to manipulate CSV files
#different mathmatical functions:
import datetime as dtm
import math
from datetime import datetime
import os
import pandas as pd
def roundSeconds(dateTimeObject):
    newDateTime = dateTimeObject + dtm.timedelta(seconds=.5)
    newnewDateTime=newDateTime.replace(microsecond=0)
    offset = newnewDateTime.second % 10
    if offset < 5:
        rounded = newnewDateTime - dtm.timedelta(seconds=(offset))
    else:
        rounded = newnewDateTime + dtm.timedelta(seconds=((10-offset)))
    return rounded
#developing behavior dictionary
def dictionary(directory, j, k):
    temp={}
    behavior={}
    # j=0
    # k=0
    for root, dirs, files in os.walk(directory):
        for file in files:
            # print(file)
            file=open(os.path.join(directory, file),'r')
            csvreader = csv.reader(file)
            header=next(csvreader)
            for h in header:
                if h=="Time":
                    v=header.index(h)
            for row in csvreader:
                name=os.path.basename(file.name)[:17]
                date_time=datetime.strptime(name, '%Y-%m-%d_%H%M%S')
                if row[10]=='drinking':
                    n=math.floor(j/2)
                    datetimeobject=date_time+dtm.timedelta(seconds=float(row[v]))
                    newdatetimeobject=roundSeconds(datetimeobject)
                    behavior[f'd{j}']=newdatetimeobject
                    j+=1
                elif 'feeding:' in row[10]:
                    l=math.floor(k/2)
                    datetimeobject=date_time+dtm.timedelta(seconds=float(row[v]))
                    newdatetimeobject=roundSeconds(datetimeobject)
                    behavior[f'f{k}']=newdatetimeobject
                    k+=1
            file.close()
    return behavior, j,k

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    