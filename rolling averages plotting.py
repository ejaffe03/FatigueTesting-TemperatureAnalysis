# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 10:06:20 2024

@author: ellie
"""
## Main script to run for data analysis of MiGUT data


#%% Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import math
import matplotlib as mpl
#from pathlib import Path
from Plot_EGG import *
import Plot_EGG_20230924 as s
from AG_Analysis_Funcs import *
from prep_behavior_data import *
from datetime import datetime
from behavior_analysis_data import *
import datetime as dtm
import matplotlib.dates as mdates
from scipy.stats import spearmanr
mpl.rc("figure", dpi=600)
# from sklearn.metrics import r2_score 


#%% Input/output files for temp data

output_type = '.png'
output_folder = "../documentation/figures/"


animals=['fennec','fennec', 'capybara', 'capybara']
days=['f1','f2','c1','c2']
start_datetimes=['2023-10-10 00:00:00.00','2023-10-19 00:00:00.00','2023-09-30 00:00:00.00','2023-10-03 00:00:00.00']
end_datetimes=['2023-10-10 23:59:59.59','2023-10-19 23:59:59.59','2023-09-30 23:59:59.59','2023-10-03 23:59:59.59']

#%% Functions to read the temp data files and return a dataframe per each date input. This has a rolling time period (change here) and returns per 10 seconds- a 1 or 0
#depending on if there is a temp drop or not
def single_mode(series):
    multimodes = pd.Series.mode(series)
    # print(multimodes)
    return np.max(multimodes)

def read_files(animal, day, start_datetime, end_datetime):
    temp_files = sorted(glob.glob(f"C:/Users/ellie/Downloads/tempdata/temp_data/*{animal}*.txt")) #os.listdir('../data/')
    # print(temp_files)
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
    # data0 = data0.mask(data0["Channel 0"] > 38)
    mask = data0['Channel 0'] == 40.0
    data1=data0[mask]
    data0 = data0[~mask]
    data1 = data1[data1['Channel 0'].notna()]
    data0 = data0[data0['Channel 0'].notna()]
    mask = data0['Channel 0'] == 39.0
    data1=data0[mask]
    data0 = data0[~mask]
    data1 = data1[data1['Channel 0'].notna()]
    data0 = data0[data0['Channel 0'].notna()]
    data0.index.name = 'Date'
    data0["out"]=1
    data1['out']=0
    data0=pd.concat([data0,data1])
    # print(data0)
    
    data0=data0.drop(columns=['timestamp_temp','rssi','Channel 0', 'beh_label'])
    data0=data0.sort_index()
    data1=data0
    # newrolling=data0.rolling('10min',center=True)
    by_10_min=data0.rolling('10min', center=True).sum()/60
    by_10_min.index=by_10_min.index.time
    date = str(datetime.strptime('2018-01-01', '%Y-%m-%d').date())
    by_10_min.index = pd.to_datetime(date + " " + by_10_min.index.astype(str))
    by_10_min=by_10_min.sort_index()
    # start_time = data0.index[0]
    # data0['time_diff'] = (data0.timestamp_temp - start_time).dt.total_seconds()/3600
    return by_10_min, data1


#%% Running all days and compiling into one dataframe-temp
i=0
data1=[]
rollingvalues=[]
while i<4:
    print(animals[i],days[i],start_datetimes[i],end_datetimes[i])
    by_10_min, data2=read_files(animals[i],days[i],start_datetimes[i],end_datetimes[i])
    rollingvalues.append(data2)
    data1.append(by_10_min)
    i+=1
data2=pd.concat(rollingvalues)
data0=pd.concat(data1, axis=1)
#%% opening behavior labelling files and returning dictionaries with events


directory = (r"C:\\Users\ellie\Downloads\UROP\Temperature Analysis\labeled data\fennec\f1")
directory1=(r"C:\\Users\ellie\Downloads\UROP\Temperature Analysis\labeled data\fennec\f2")
directory2=(r"C:\\Users\ellie\Downloads\UROP\Temperature Analysis\labeled data\capybara\c1")
directory3=(r"C:\\Users\ellie\Downloads\UROP\Temperature Analysis\labeled data\capybara\c2")
behavior1={}

s=dictionary(directory)
s1=dictionary(directory1)
s2=dictionary(directory2)
s3=dictionary(directory3)

#%%Function to use the behavior dictionary and return a dataframe per each date input. Rolling time period and returns per 10 seconds- a 1 or 0 depending on if event occuring
def behavior_probability(behavior1, i):
    l=len(behavior1.keys())-2
    # print(behavior1.keys())
    behavior2={}
    n=0
    while n<l:
        start_date=list(behavior1.values())[n]
        n+=1
        end_date=list(behavior1.values())[n]
        delta=dtm.timedelta(seconds=10)
        # timerange=timerange.to_pydatetime()
    # print(timerange)
        while (start_date<end_date):
            behavior2[start_date]=1
            start_date+=delta
        n+=1
        second_end_date=list(behavior1.values())[n]
        while (end_date<second_end_date):
            behavior2[end_date]=0
            end_date+=delta
    timerange=pd.date_range(str(pd.to_datetime(start_datetimes[i])), str(pd.to_datetime(end_datetimes[i])), freq='10s')
    for x in timerange:
        if x not in behavior2.keys():
            behavior2[x]=0
    dictionary = dict([(key, behavior2[key]) for key in sorted(behavior2)])
    df = pd.DataFrame.from_dict(dictionary, orient='index', columns=['DateValue'])
    # print(df)
    df1=df
    df=df.rolling('10min', center=True).sum()/(60)
    df.index=df.index.time
    df.sort_index()
    return df, df1

#%% Running all days and compiling into one dataframe-behavior
df1=[]
by_10_min_behavior, df2=behavior_probability(s, 0)
df1.append(df2)
by_10_min_behavior.rename(columns={'DateValue': 'b'}, inplace=True)
behaviorprob, df2 =behavior_probability(s1, 1)
df1.append(df2)
by_10_min_behavior=pd.concat([by_10_min_behavior,behaviorprob], axis=1)
behaviorprob, df2 =behavior_probability(s2, 2)
df1.append(df2)
by_10_min_behavior=pd.concat([by_10_min_behavior,behaviorprob], axis=1)
behaviorprob, df2 =behavior_probability(s3, 3)
df1.append(df2)
by_10_min_behavior=pd.concat([by_10_min_behavior,behaviorprob], axis=1)
df1=pd.concat(df1)

#%%Behavior- compiling all days into 1 day and calculating average and std and normalize

by_10_min_behavior=by_10_min_behavior.fillna(0)
by_10_min_behavior['e'] = by_10_min_behavior.sum(axis=1, numeric_only=True)

by_10_min_behavior['e'] =by_10_min_behavior['e']/4
by_10_min_behavior['std']=by_10_min_behavior['DateValue'].std(axis=1)/2
by_10_min_behavior=by_10_min_behavior.sort_index()
by_10_min_behavior1=by_10_min_behavior.drop(columns=['DateValue', 'std'])

date = str(datetime.strptime('2018-01-01', '%Y-%m-%d').date())
by_10_min_behavior.index = pd.to_datetime(date + " " + by_10_min_behavior.index.astype(str))
by_10_min_behavior1.index = pd.to_datetime(date + " " + by_10_min_behavior1.index.astype(str))
# by_10_min_behavior=by_10_min_behavior.resample('15min').sum()/90
# plt.scatter(by_10_min_behavior.index,by_10_min_behavior['e'])
# myFmt = mdates.DateFormatter('%H:%M')
# plt.gca().xaxis.set_major_formatter(myFmt)
#%% Temp-compiling all days into 1 day and calculating average and std and normalize
data0=data0.fillna(method='ffill', limit=120)    
data0=data0.fillna(method='bfill', limit=120)    
data0=data0.fillna(0)
data0['e1'] = data0.sum(axis=1, numeric_only=True)
data0['e1'] =data0['e1']/4
data0['std']=data0['out'].std(axis=1)
data1=data0.drop(columns=['out','std'])
df_norm = (data1-data1.min())/(data1.max()-data1.min())
df_norm1 = (by_10_min_behavior1-by_10_min_behavior1.min())/(by_10_min_behavior1.max()-by_10_min_behavior1.min())
df_norm=df_norm.fillna(0)
correlating=pd.concat([by_10_min_behavior,data0], axis=1)
correlating=correlating.fillna(method='ffill', limit=120)
correlating=correlating.fillna(method='bfill', limit=120)
# correlating=correlating.fillna(0)
correlating.to_csv('output.csv')
# print(correlating)
#%% Rolling stats
# data2=data2.fillna(method='ffill', limit=120)    
# data2=data2.fillna(method='bfill', limit=120)    
# data2=data2.fillna(0)
# alldata=pd.concat([df1,data2], axis=1)
# alldata=alldata.fillna(0)
# alldata=alldata.sort_index()
# indexes=alldata.index.tolist()
# df=alldata.rolling('10min', center=True, method="table")
# listofdfs=[]
# correlationdict={}

# for window in df:
#     # print(window.values)
#     if not any(x!=0 for x in window['out'].values) and not any(x!=0 for x in window['DateValue'].values):
#         corre=1
#     elif not np.any(window['out'].values) or not np.any(window['DateValue'].values):
#         out=window['out'].values
#         DateValue=window['DateValue'].values
#         numer=6*sum((out[i]-DateValue[i])**2 for i in range(0, len(out)))
#         den=len(out)*(len(out)**2-1)
#         corre=1-numer/den
#         if corre>.97:
#             corre=0
#     else:
#         corre,_=spearmanr(window['out'].values, window['DateValue'].values)
#     if corre<0:
#         corre=0
#     windowindex=window.index.tolist()[0]
#     windowindex=windowindex.time()
#     date = str(datetime.strptime('2018-01-01', '%Y-%m-%d').date())
#     windowindex = pd.to_datetime(date + " " + str(windowindex))
#     if windowindex in correlationdict:
#         origvalue=correlationdict.get(windowindex)
#         if pd.isna(origvalue):
#             origvalue=1
#         correlationdict[windowindex]=(origvalue+corre)/2
#     else:
#         correlationdict[windowindex]=corre
# dictionary = dict([(key, correlationdict[key]) for key in sorted(correlationdict)])

# df3=pd.DataFrame.from_dict(dictionary, orient='index', columns=['correlation'])
# df = pd.DataFrame({'Date':pd.date_range('2018-01-01 0:00:00','2018-01-01 23:59:59', freq='10s'), 'Const':1})
# df.index=df['Date']
# df3=pd.concat([df3,df])
# df3=df3.drop(columns=['Const','Date'])
# # df3.index+=pd.DateOffset(minutes=5)
# # df3=df3.fillna(method='ffill', limit=960)
# # df3=df3.fillna(method='bfill', limit=960)
# # df3=df3.fillna(method='ffill', limit=960)
# # df3=df3.fillna(method='bfill', limit=960)
# df3=df3.dropna()
# df3=df3.sort_index()
# df3=df3.rolling('10min', center=True).apply(lambda x: x.sum()/len(x))

#%% Pearson's coefficient/calculating correlation coefficient
corre = correlating.corr(method='spearman')
# corre2=correlating.corr(method='pearson')
# corre1,_=spearmanr(correlating.values)
# correlating['e1']=correlating['e1'].shift(-5)
correlating=correlating.fillna(0)
# r2 = r2_score(correlating['e1'], correlating['e'])
# print(r2) 
import statsmodels.api as sm

statsmodels_correlation = sm.OLS(correlating['e'],correlating['e1']).fit().rsquared
print('Statsmodels Correlation:', statsmodels_correlation)
numpy_correlation = np.corrcoef(correlating['e'], correlating['e1'])[0, 1]
print('NumPy Correlation:', numpy_correlation)
from scipy.stats import pearsonr
scipy_correlation, _ = pearsonr(correlating['e'], correlating['e1'])
print('SciPy Correlation:', scipy_correlation)
def mean(arr):
    return sum(arr) / len(arr)
# function to calculate cross-correlation
def cross_correlation(x, y):
    # Calculate means
    x_mean = mean(x)
    y_mean = mean(y)
    
    # Calculate numerator
    numerator = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y))
    
    # Calculate denominators
    x_sq_diff = sum((a - x_mean) ** 2 for a in x)
    y_sq_diff = sum((b - y_mean) ** 2 for b in y)
    denominator = math.sqrt(x_sq_diff * y_sq_diff)
    correlation = numerator / denominator
    return correlation
correlation = cross_correlation(correlating['e'], correlating['e1'])
print('Correlation:', correlation)
#%% plotting
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = "9"
fig = plt.figure()

gs = fig.add_gridspec(2, hspace=0)
axs = gs.subplots(sharex=True, sharey=True)
axs[0].plot(by_10_min_behavior.index, by_10_min_behavior['e'], label="Probability of Behavior Event", color='k',linewidth=.5)
axs[0].fill_between(by_10_min_behavior.index, by_10_min_behavior['e']-by_10_min_behavior['std'], by_10_min_behavior['e']+by_10_min_behavior['std'], alpha=.2)
axs[1].plot(data0.index, data0['e1'], label="probability of Temperature Drop", color='k',linewidth=.5)
axs[1].fill_between(data0.index, data0['e1']-data0['std']/2, data0['e1']+data0['std']/2, alpha=.2)
ax=axs.flat
ax[0].set(ylabel='$P_{Ingestion}$')
ax[1].set(ylabel='$P_{Temp Drop}$')
ax[1].set(xlabel='Time of Day')
# ax[2].set(ylabel='Spearman Value')
axs[0].set_ylim(0,1)
axs[1].set_ylim(0,1)

# fig.suptitle('Probability of Behavior and Temperature Events Over the Course of the Day')
axs[0].axvspan(pd.to_datetime('2018-01-01 0:00:00'), pd.to_datetime('2018-01-01 7:00:00'), color='grey', alpha=0.3, lw=0)
axs[0].axvspan(pd.to_datetime('2018-01-01 19:00:00'), pd.to_datetime('2018-01-01 23:59:59'), color='grey', alpha=0.3, lw=0)
axs[1].axvspan(pd.to_datetime('2018-01-01 0:00:00'), pd.to_datetime('2018-01-01 7:00:00'), color='grey', alpha=0.3, lw=0)
axs[1].axvspan(pd.to_datetime('2018-01-01 19:00:00'), pd.to_datetime('2018-01-01 23:59:59'), color='grey', alpha=0.3, lw=0)
# axs[2].axvspan(pd.to_datetime('2018-01-01 0:00:00'), pd.to_datetime('2018-01-01 7:00:00'), color='grey', alpha=0.3, lw=0)
# axs[2].axvspan(pd.to_datetime('2018-01-01 19:00:00'), pd.to_datetime('2018-01-01 23:59:59'), color='grey', alpha=0.3, lw=0)
# axs[2].plot(df3.index,df3['correlation'], color='k', label='Spearman rho value', linewidth=.5)
# axs[2].hlines(y=df3.mean(),xmin=pd.to_datetime('2018-01-01 0:00:00'), xmax= pd.to_datetime('2018-01-01 23:59:59'), linestyle='--',linewidth=.5)
myFmt = mdates.DateFormatter('%H:%M')
plt.gca().xaxis.set_major_formatter(myFmt)
plt.show()
#%% T score and P value
from scipy.stats import t

def calculate_t_score(sample1, sample2):
    mean1 = np.mean(sample1)
    mean2 = np.mean(sample2)
    std1 = np.std(sample1, ddof=8638) 
    std2 = np.std(sample2, ddof=8638)
    n1 = len(sample1)-2
    n2 = len(sample2)-2
 
    t_score = (corre)/np.sqrt((1-corre**2)/8638)
    return abs(t_score)
 
# Step 2: Determine the degrees of freedom (df)
def calculate_degrees_of_freedom(sample1, sample2):
    n1 = len(sample1)
    n2 = len(sample2)
    df = n1 + n2 - 4  # For a two-sample t-test
    return df/2
 
# Step 3: Identify the appropriate t-distribution
# (The scipy.stats.t distribution is used, which automatically considers the degrees of freedom)
 
# Step 4: Find the p-value
def calculate_p_value(t_score, df):
    p_value = 2 * (1 - t.cdf(np.abs(t_score), df))
    return p_value

t_score=calculate_t_score(correlating['e'],correlating['e1'])
df=calculate_degrees_of_freedom(correlating['e'],correlating['e1'])
p_value=calculate_p_value(t_score,df)
print(t_score,df,p_value)

bins=np.arange(0,1,0.05)
# plt.hist(correlating['e'], bins)
# plt.show()
# plt.hist(correlating['e1'],bins)
# plt.show()
# plt.scatter(correlating['e'], correlating['e1'])
# plt.show()