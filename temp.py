# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#insert desired times:
start_time1='9/28/2023 10:25 AM'
end_time1='9/28/2023 10:35 AM'


import matplotlib.pyplot as plt #for created plots
import matplotlib as mpl
mpl.rc("figure", dpi=200)
import csv # this gives us tools to manipulate CSV files
#different mathmatical functions:
from datetime import datetime

def nearest(items, pivot):
    return min(items, key=lambda x: abs(datetime.strptime(x[2]+" "+x[6], '%m/%d/%Y %I:%M:%S %p') - pivot))
def nearest1(items, pivot):
    return min(items, key=lambda x: abs(datetime.strptime(x[20], '%m/%d/%Y %H:%M:%S') - pivot))



file = open(r"C:\\Users\ellie\Downloads\temp_plot_fennec.csv")
file1 = open(r"C:\\Users\ellie\Downloads\parsed data - Fennec.csv")
data=[]
deltat=0
csvreader = csv.reader(file)
next(csvreader)
for row in csvreader:
            #time automatically starts at 0 for each test- this increases the time so that it counts
            #total time the device is run
            row[0]=(row[0])
            data.append(row)
file.close()
newnumber=data[-1]
deltat=newnumber[0]
subdata=data[0:len(data)]


data1=[]
deltat1=0
csvreader = csv.reader(file1)
next(csvreader)
next(csvreader)
for row in csvreader:
            #time automatically starts at 0 for each test- this increases the time so that it counts
            #total time the device is run
            row[0]=(row[0])
            data1.append(row)
file1.close()
newnumber1=data1[-1]
deltat1=newnumber1[0]
subdata1=data1[0:len(data)]
start_time=datetime.strptime(start_time1, '%m/%d/%Y %I:%M %p')
end_time=datetime.strptime(end_time1, '%m/%d/%Y %I:%M %p')
y=subdata1.index(nearest(subdata1,start_time))
k=subdata1.index(nearest(subdata1,end_time))
x=subdata.index(nearest1(subdata,start_time))
f=subdata.index(nearest1(subdata,end_time))

n1=k-y
n=f-x
timedate=[None]*n
temp=[None]*n
i=0
while i<n:
    timedate[i] = datetime.strptime(subdata[i+x][20], '%m/%d/%Y %H:%M:%S')
    # timedate[i]=datetime(subdata[i][20]) # times at which each of the samples were taken
    temp[i]=subdata[i+x][23] # first channel of Digilent Discovery 2
    i=i+1

i=0
activity=[]
# print(subdata1)
while i<n1:
    if subdata1[i+y][5]=='drinking':
        plt.axvspan(datetime.strptime(subdata1[i+y][2]+" "+subdata1[i+y][6], '%m/%d/%Y %I:%M:%S %p'), datetime.strptime(subdata1[i+y][2]+" "+subdata1[i+y][7], '%m/%d/%Y %I:%M:%S %p'), 
                    color='red', alpha=0.25)
        activity.append(1)

    elif subdata1[i+y][5]=='eating':
        plt.axvspan(datetime.strptime(subdata1[i+y][2]+" "+subdata1[i+y][6], '%m/%d/%Y %I:%M:%S %p'), datetime.strptime(subdata1[i+y][2]+" "+subdata1[i+y][7], '%m/%d/%Y %I:%M:%S %p'), 
                    color='blue', alpha=0.25)
        activity.append(2)

    i=i+1

plt.plot(timedate, temp)
plt.gcf().autofmt_xdate()
plt.gca().invert_yaxis()
plt.xlabel('Time')
plt.ylabel('Temp (C)')
plt.title('Temperature of the stomach while drinking')
if activity[0]==1:
    if 2 in activity:
        plt.legend(['drinking','eating'], loc="lower right")
    else:
        plt.legend(['drinking'], loc="lower right")
elif activity[0]==2:
    if 1 in activity:
        plt.legend(['eating','drinking'], loc="lower right")
    else:
        plt.legend(['eating'], loc="lower right")
plt.show()