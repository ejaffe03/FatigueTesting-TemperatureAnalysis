## Main script to run for data analysis of MiGUT data


## Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#import pickle
import glob
#from tqdm.notebook import tqdm
#from pathlib import Path
import scipy
from Plot_EGG import *
import Plot_EGG_20230924 as s
from AG_Analysis_Funcs import *
from prep_behavior_data import *
from datetime import datetime
from behavior_analysis_data import *
import datetime as dtm


#%%

animals=['fennec','fennec', 'capybara', 'capybara']
days=['f1','f2','c1','c2']
start_datetimes=['2023-10-10 00:00:00.00','2023-10-19 00:00:00.00','2023-09-30 00:00:00.00','2023-10-03 00:00:00.00']
end_datetimes=['2023-10-10 23:59:59.59','2023-10-19 23:59:59.59','2023-09-30 23:59:59.59','2023-10-03 23:59:59.59']

window_length = 10


#%% Reading temperature data and returning dataframe with data
def single_mode(series):
    multimodes = pd.Series.mode(series)
    # print(multimodes)
    return np.max(multimodes)

most_freq_val = lambda x: scipy.stats.mode(x)[0][0]

def read_files(animal, day, start_datetime,end_datetime):
    temp_files = sorted(glob.glob(f"C:/Users/ellie/Downloads/tempdata/temp_data/*{animal}*.txt")) #os.listdir('../data/')
    data0=read_egg_v3_temp_list(temp_files, scale=600, interp=0, rate=1/10)

    behavior_files = sorted(glob.glob((f"../data_labeling/labeled data/{animal}/{day}/*.csv")))
    data1 = read_behavior_data(behavior_files, pd.to_datetime(start_datetime), pd.to_datetime(end_datetime), header=0, rate=0, scale=0, error=0, interp=0)
    data0["timestamps_days"] = data0["timestamps"]/60/60/24
    data0["rssi"] = data0["rssi"].astype(int)
    bad_times = data0.loc[data0["timestamps_days"].diff() > 0.25]
# # print(bad_times)
# #%% reduce down to single day
    mask_d0 = (data0['timestamp_temp'] > start_datetime) & (data0['timestamp_temp'] <= end_datetime)
    data0 = data0.loc[mask_d0]
    data0 = pd.DataFrame(data0.groupby([pd.Grouper(key='timestamp_temp', freq='10s')]).agg({"Channel 0": [single_mode], "rssi": [single_mode]})).reset_index()
    data0.columns = data0.columns.get_level_values(0)
    data0.set_index('timestamp_temp', inplace=True, drop=False)
    data0 = data0[['timestamp_temp', 'Channel 0', 'rssi']]

    data0['beh_label'] = 0
    
    data1['start_time'] = pd.to_datetime(data1['start_time'], format='%Y-%m-%d %H:%M:%S')
    data1['end_time'] = pd.to_datetime(data1['end_time'], format='%Y-%m-%d %H:%M:%S')

# merge dataframes
    for i, r in data1.iterrows():
        for idx, row in data0.iterrows():
            if (r["start_time"] < row["timestamp"]) & (r["end_time"] >= row["timestamp"]):
                data0.at[idx, "beh_label"] = r.behavior
            else: pass
    return data0
i=0
data1=[]
while i<4:
    data1.append(read_files(animals[i],days[i],start_datetimes[i],end_datetimes[i]))
    i+=1

data0=pd.concat(data1)
#%% Removing temp data for not temp drops and na values then dividing by temperature drop
indices=data0.index.to_pydatetime()
indices2=indices.tolist()
indixes=data0['Channel 0'].tolist()

mask = data0['Channel 0'] >= 40.0
data0 = data0[~mask]
data0 = data0[data0['Channel 0'].notna()]
mask = data0['Channel 0'] == 39.0
data0 = data0[~mask]
data0 = data0[data0['Channel 0'].notna()]
data0.index = pd.to_datetime(data0.index)

data0['tvalue'] = data0.index
data0['delta'] = (data0['tvalue']-data0['tvalue'].shift()).fillna(dtm.timedelta(minutes=0))
indexes=np.where(data0['delta']>dtm.timedelta(seconds=90))
l_mod = [0] + list(indexes[0]) + [max(list(indexes[0]))+1]

list_of_dfs = [data0.iloc[l_mod[n]:l_mod[n+1]] for n in range(len(l_mod)-1)]

#%% Parsing behavior data and Creating dictionaries for TP, FN and FP
directory = (r"C:\\Users\ellie\Downloads\labeled data\fennec\f1")
directory1=(r"C:\\Users\ellie\Downloads\labeled data\fennec\f2")
directory2=(r'C:\\Users\ellie\Downloads\labeled data\capybara\c1')
directory3=(r'C:\\Users\ellie\Downloads\labeled data\capybara\c2')
behavior1={}
j=0
k=0
s, j, k=dictionary(directory, j, k)
behavior1.update(s)
s, j, k=dictionary(directory1, j, k)
behavior1.update(s)
s, j, k=dictionary(directory2, j, k)
behavior1.update(s)
s, j, k=dictionary(directory3, j, k)
behavior1.update(s)
l=len(behavior1.keys())-2
indices=data0.index.to_pydatetime()
behavior={}
events={}
temp=[]
n=0
while n<l:
    start_date=list(behavior1.values())[n]
    output=list(behavior1.values())[n]
    n+=1
    end_date=list(behavior1.values())[n]
    timerange=pd.date_range(start_date, end_date+dtm.timedelta(minutes=15), freq="10s", inclusive='both') #start and end date
    timerange=timerange.to_pydatetime() #creates time range of event
    newmask=np.in1d(indices,timerange) #creates array of bool of overlap between time event and temp drops
    if newmask.any(): #if any overlap
        newindex=indices[np.where(np.array(newmask))[0]]
        array=[np.in1d(newindex,y).any() for y in list_of_dfs]
        if len((np.where(array)[0]).tolist())>0:
            # print((np.where(array)[0]).tolist()[0])
            temp.append(list_of_dfs[(np.where(array)[0]).tolist()[0]])
            list_of_dfs.pop((np.where(array)[0]).tolist()[0])
        # print(temp)
        delta=dtm.timedelta(seconds=10)
        while (start_date<end_date):
            behavior[start_date]=1
            # events[start_date]=0
            start_date+=delta
        n+=1
        second_end_date=list(behavior1.values())[n]
        while (end_date<second_end_date):
            # events[end_date]=0
            end_date+=delta
    else:
        delta=dtm.timedelta(seconds=10)
        while (start_date<end_date):
            behavior[start_date]=0
            events[start_date]=1
            start_date+=delta
        n+=1
        second_end_date=list(behavior1.values())[n]
        while (end_date<second_end_date):
            # events[end_date]=0
            end_date+=delta
            
#%%Turning dictionaries into dataframes transferring data types for later processes
truepos=pd.concat(temp)
truepos['out']=1
truepos.index=pd.to_datetime(truepos.index)
truepos=truepos['out']
falsepos=pd.concat(list_of_dfs)
falsepos['out']=1
falsepos.index=pd.to_datetime(falsepos.index)
falsepos=falsepos['out']
print(falsepos)
dictionary = dict([(key, behavior[key]) for key in sorted(behavior)])
truepos1 = pd.DataFrame.from_dict(dictionary, orient='index', columns=['DateValue'])
truepos1.index.name = 'Date'
truepos1['out']=1
truepos1.index=pd.to_datetime(truepos1.index)
truepos1=truepos1['out']

dictionary = dict([(key, events[key]) for key in sorted(events)])
falseneg = pd.DataFrame.from_dict(dictionary, orient='index', columns=['DateValue'])
falseneg.index.name = 'Date'
falseneg['out']=1
falseneg.index=pd.to_datetime(falseneg.index)
falseneg=falseneg['out']
date = str(datetime.strptime('2018-01-01', '%Y-%m-%d').date())
truepos=truepos.resample('10s').sum()

falsepos.index=falsepos.index.time
falsepos.index = pd.to_datetime(date + " " + falsepos.index.astype(str))
falsepos=falsepos.sort_index()
falsepos=falsepos.resample('60min').sum()/4

falseneg.index=falseneg.index.time
falseneg.index = pd.to_datetime(date + " " + falseneg.index.astype(str))
falseneg=falseneg.sort_index()
falseneg=falseneg.resample('60min').sum()/4
truepos1=truepos1.resample('10s').sum()
truepos=pd.concat([truepos,truepos1])

truepos.index=truepos.index.time
truepos.index = pd.to_datetime(date + " " + truepos.index.astype(str))
truepos=truepos.sort_index()
truepos=truepos.resample('60min').sum()/8

#%% Creating dataframe for TN values
timerange=pd.date_range(str(pd.to_datetime('2018-01-01 0:00:00')), str(pd.to_datetime('2018-01-01 23:59:59')), freq='10min')
matrix=pd.concat([truepos,falsepos,falseneg], axis=1)
matrix=matrix.fillna(0)
dic={}
for x in range(0,len(timerange)):
    #-falsepos[falsepos.index==timerange[x]].values-falseneg[falseneg.index==timerange[x]].values
    newlist=matrix[matrix.index==timerange[x]].values.tolist()
    if newlist:
        # print(newlist)
        dic[timerange[x]]=360-newlist[0][0]-newlist[0][1]-newlist[0][2]
# print(dic)
dictionary = dict([(key, dic[key]) for key in sorted(dic)])
trueneg = pd.DataFrame.from_dict(dictionary, orient='index', columns=['DateValue'])
#%% Plotting TP, FP, FN, and TN
# print(matrix)
# plt.xlabel('Time of Day')
# plt.ylabel('Average Number of FN and FP')
# plt.title('Average Number of False Negatives and False Positives Over the Course of the Day')
# plt.scatter(truepos.index,truepos, label='True Positives')
# plt.scatter(falsepos.index,falsepos, label='False Positives')
# plt.scatter(falseneg.index,falseneg, label='False Negative')
# plt.scatter(trueneg.index,trueneg['DateValue'], label='True Negative')

# plt.legend(loc='upper right')
# plt.gcf().autofmt_xdate()
# plt.plot() , cmap='Blues', fmt='d'
#%% Resampling for 24 hours, turning into percents and then displaying confusion matrix

truepos=truepos.resample('24h').sum()
trueneg=trueneg.resample('24h').sum()
falsepos=falsepos.resample('24h').sum()
falseneg=falseneg.resample('24h').sum()
truenegvalue=trueneg.values[0].item()
falsenegvalue=falseneg.values[0].item()
falseposvalue=falsepos.values[0].item()
trueposvalue=truepos.values[0].item()
truenegper=(truenegvalue/(truenegvalue+falsenegvalue))
falsenegper=(falsenegvalue/(truenegvalue+falsenegvalue))
falseposper=(falseposvalue/(falseposvalue+trueposvalue))
trueposper=(trueposvalue/(trueposvalue+falseposvalue))
plt.rcParams["font.size"] = "9"

cm_data = [[truenegper, falsenegper], [falseposper,trueposper]]
print(cm_data)
s=sns.heatmap(cm_data, annot=True, cmap='Blues', xticklabels=['No Behavior', 'Behavior'], yticklabels=['No Drop', 'Drop'], fmt='.1%')
s.set(xlabel='Behavior Event', ylabel='Temperature Drop')
plt.plot()