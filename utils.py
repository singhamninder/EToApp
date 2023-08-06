import streamlit as st
import pandas as pd
import numpy as np
import pyeto
import matplotlib.pyplot as plt
import pickle

def plot_results(df, model):
    fig, ax = plt.subplots()
    if 'Eto' in df.columns:
        ax.plot([d.timetuple().tm_yday for d in df.Date], df['Eto'], 'o', label = '$ET_{o}$')
        ax.plot([d.timetuple().tm_yday for d in df.Date], df['EThs'], linewidth=2, label = '$ET_{HS}$')
        ax.plot([d.timetuple().tm_yday for d in df.Date], df[model], linewidth=2, ls ='--', label = model)
        ax.legend(prop={'size': 13})
    else:
        ax.plot([d.timetuple().tm_yday for d in df.Date], df['EThs'], linewidth=2, label = '$ET_{HS}$')
        ax.plot([d.timetuple().tm_yday for d in df.Date], df[model], linewidth=2, ls ='--', label = model)
        ax.legend(prop={'size': 13})
    fig.supxlabel('DOY', size = 14)
    fig.supylabel('$ET{_{o}} [mm d{^{-1}}$]', size = 16, weight='bold')
    return fig

def split_sequences(sequences, n_steps):
    '''Function to convert input timeseries to lag observations .i.e multiple samples with specified number of timesteps'''
    X= []
    for i in range(len(sequences)):
        end_ix = i + n_steps
        if end_ix > len(sequences):
            break
        seq_x = sequences[i:end_ix, :]
        X.append(seq_x)
    return np.array(X)