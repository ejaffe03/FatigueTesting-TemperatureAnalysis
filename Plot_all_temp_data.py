import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import math
import glob
import matplotlib as mpl
from tqdm.notebook import tqdm
from datetime import datetime
from os.path import exists
from scipy import signal
from pathlib import Path
import scipy
import time
from scipy import stats
from Plot_EGG import *
import Plot_EGG_20230924 as s
from AG_Analysis_Funcs import *
from prep_behavior_data import *
from datetime import datetime

#%%
def single_mode(series):
    multimodes = pd.Series.mode(series)
    # print(multimodes)
    return np.max(multimodes)

def preprocess_temp_data(file):
    data0 = read_egg_v3_temp_list(file, scale=600, interp=0, rate=1/10)
    data0['timestamp_temp'] = pd.to_datetime(data0['timestamp_temp'], format='%d%m%Y:%H:%M:%S')
    data0['date'] = data0.timestamp_temp.dt.date
    data0['time'] = data0.timestamp_temp.dt.time
    data0['timestamps_days'] = (data0['date'] - data0['date'][0])
    data0['timestamps_days'] = pd.to_timedelta(data0['timestamps_days'])

    # data0["timestamps_days"] = data0["timestamps"]/60/60/24
    data0["rssi"] = data0["rssi"].astype(int)
    # bad_times = data0.loc[data0["timestamps_days"].diff() > 0.25]

    data0 = pd.DataFrame(data0.groupby([pd.Grouper(key='timestamp_temp', freq='10s')]).agg({"Channel 0": [single_mode], "rssi": [single_mode], "timestamps_days": [single_mode]})).reset_index()
    data0.columns = data0.columns.get_level_values(0)
    data0.set_index('timestamp_temp', inplace=True, drop=False)
    data0 = data0[['timestamp_temp', 'Channel 0', 'rssi', 'timestamps_days']]
    # data0['timestamps_days'] = data0['timestamps_days'].ffill().astype(int)
    data0.to_csv(preprocessedData_file)
    return data0

def plot_data(pp_file):
    dataset = pd.read_csv(pp_file)
    dataset.drop(columns={'timestamp_temp.1'}, inplace=True)
    dataset['timestamp_temp'] = pd.to_datetime(dataset['timestamp_temp'])
    dataset['time'] = dataset.timestamp_temp.dt.time

    for i in range(87):
        d = dataset[dataset['timestamps_days'] == f'{i} days']
        print(d['Channel 0'].dtype)
        sns.lineplot(x=d['time'].astype(str), y=d['Channel 0'])
        plt.savefig(f"../documentation/figures/large/fennec_temp_{i}.svg")
    # plt.show()
    # Initialize the FacetGrid object
    # pal = sns.cubehelix_palette(87, rot=-.25, light=.7)
    # g = sns.FacetGrid(dataset, row=dataset["timestamps_days"], hue=dataset["timestamps_days"], aspect=15, height=.5, palette=pal)

    # g.map(sns.lineplot, data=dataset, y=dataset['Channel 0'], x=dataset['time'])
    # print(dataset)
    # data = []

    # for day, group in dataset.groupby('timestamps_days'):
    #     print(day)

    

    # print(dataset)


    # #     # Create the data
    # rs = np.random.RandomState(1979)
    # x = rs.randn(500)
    # g = np.tile(list("ABCDEFGHIJ"), 50)
    # df = pd.DataFrame(dict(x=x, g=g))
    # m = df.g.map(ord)
    # df["x"] += m

    # # Initialize the FacetGrid object
    # pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
    # g = sns.FacetGrid(df, row="g", hue="g", aspect=15, height=.5, palette=pal)

    # # Draw the densities in a few steps
    # g.map(sns.lineplot)
    # g.map(sns.kdeplot, "x",
    #     bw_adjust=.5, clip_on=False,
    #     fill=True, alpha=.5, linewidth=1.5)
    # g.map(sns.kdeplot, "x", clip_on=False, color="g", lw=2, bw_adjust=.5)

    # # passing color=None to refline() uses the hue mapping
    # g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)


    # # Define and use a simple function to label the plot in axes coordinates
    # def label(x, color, label):
    #     ax = plt.gca()
    #     ax.text(0, .2, label, fontweight="bold", color=color,
    #             ha="left", va="center", transform=ax.transAxes)

    # g.map(label, "x")

    # # Set the subplots to overlap
    # g.figure.subplots_adjust(hspace=-.25)

    # # Remove axes details that don't play well with overlap
    # g.set_titles("")
    # g.set(yticks=[], ylabel="")
    # g.despine(bottom=True, left=True)
    # plt.show()


if __name__ == "__main__":
    #%%

    output_type = '.png'
    output_folder = "../documentation/figures/"
    preprocessedData_file = "../documentation/preprocessed data/temp_consolidated.txt"

    # fennec day 1
    # animal = 'fennec'
    # day = 'f1'
    # start_datetime = '2023-10-10 00:00:00.00'
    # end_datetime = '2023-10-10 23:59:59.59'

    # fennec day 2
    # animal = 'fennec'
    # day = 'f2'
    # start_datetime = '2023-10-19 00:00:00.00'
    # end_datetime = '2023-10-19 23:59:59.59'

    # capybara day 1
    # animal = 'capybara'
    # day = 'c1'
    # start_datetime = '2023-09-30 00:00:00.00'
    # end_datetime = '2023-09-30 23:59:59.59'

    # capybara day 2
    animal = 'fennec'
    day = 'c2'
    start_datetime = '2023-10-03 00:00:00.00'
    end_datetime = '2023-10-03 23:59:59.59'

    window_length = 10

    most_freq_val = lambda x: scipy.stats.mode(x)[0][0]

    # def fixrssi(col):
    #     return pd.DataFrame([x if isinstance(x, tuple) else (x, ) 
    #                         for x in col]).fillna(0).astype(int)

    temp_files = sorted(glob.glob(f"../data/temp_data/*{animal}*.txt")) #os.listdir('../data/')
    dataset = preprocess_temp_data(temp_files)
    
    plot_data(preprocessedData_file)

