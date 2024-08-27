## Imports
import matplotlib.pyplot as plt
#import seaborn as sns
import pandas as pd
import numpy as np
# from tqdm import tqdm
import os
import datetime
import glob
from os import listdir
from os.path import isfile, join
import scipy.stats as st
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
#from sklearn import preprocessing
# import mchmm as mc

import warnings
warnings.filterwarnings('ignore')


def read_behavior_data(folder, start_date, end_date, header=0, rate=14.93, scale=0, error=0, interp=0):
    beh = pd.DataFrame()
    data = []
    for file in folder:
        filename = file.split('/')[-1]
        date = filename.split("_")[0]
        time = filename.split("_")[1]
        timestamp = pd.to_datetime(f'{date} {time[:2]}:{time[2:4]}:{time[4:]}')
        prev_time = 0
        prev_behaviorlabel = 0
        prev_day = 0
        # data = []
        events = {}

        if (timestamp > start_date) & (timestamp <= end_date):
            df = pd.read_csv(file)
            df['timestamp'] = timestamp + df["Time"].apply(lambda x: datetime.timedelta(seconds=x))
            df['seconds'] = df['timestamp'].dt.second
            df['minutes'] = df['timestamp'].dt.minute
            df['hour'] = time[:2]
            df['day'] = date
            df['timestamp'] = df['day'].astype('string') + " " + df['hour'].astype('string') + ":" + df['minutes'].astype('string') + ":" + df['seconds'].astype('string')
            df = df.rename(columns={'Behavioral category': 'beh_label', 'Behavior type': 'state'})
            df['beh_label'] = df['beh_label'].str.title() 

            for idx, row in tqdm(df.iterrows()):
                curr_behaviorlabel = row.beh_label
                state = row.state

                if idx == df.index[-1]:
                    events["end_time"] = row.timestamp
                    data.append(events)
                else:
                    if (state == 'START'):
                        events["behavior"] = curr_behaviorlabel
                        events["start_time"] = row.timestamp
                        events["tod"] = row.hour
                    elif (state == 'STOP'):
                        events["end_time"] = row.timestamp
                        data.append(events)
                        events = {}
                    else:
                        pass

    data = pd.DataFrame(data, columns=['behavior', 'start_time', 'tod', 'end_time'])
    data = data.dropna()
    # print(data)
    
    consolidated_data = []
    consolidated_event = {}
    prev_label = 0
    prev_endtime = 0
    for idx, row in data.iterrows():
        curr_label = row.behavior
        if idx == data.index[-1]:
            # print('end of file')
            consolidated_event["end_time"] = row.end_time
            consolidated_data.append(consolidated_event)
            consolidated_event = {}
        elif idx == 0:
            # print(f'starting from 0 at time {row.start_time}')
            consolidated_event['behavior'] = row.behavior
            consolidated_event['tod'] = row.tod
            consolidated_event['start_time'] = row.start_time
            prev_label = curr_label
            prev_endtime = row.end_time            
        elif (curr_label != prev_label):
            # print(f'ending behavior at: {prev_endtime}')
            consolidated_event['end_time'] = prev_endtime
            # print(consolidated_event)
            consolidated_data.append(consolidated_event)
            consolidated_event = {}
            # print(f'starting new behavior at: {row.start_time}')
            consolidated_event['behavior'] = row.behavior
            consolidated_event['tod'] = row.tod
            consolidated_event['start_time'] = row.start_time
            prev_label = curr_label
            prev_endtime = row.end_time
        else:
            prev_label = curr_label
            prev_endtime = row.end_time
        # print(consolidated_data)
    

    d = pd.DataFrame(consolidated_data, columns=['behavior', 'tod', 'start_time', 'end_time'])
    # print(d)
    return d