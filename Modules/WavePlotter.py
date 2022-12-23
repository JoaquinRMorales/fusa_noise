import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torchaudio
import torchaudio.functional as functional
import torchaudio.transforms as T

from IPython.display import Audio, display, clear_output, HTML
from scipy import signal
from tkinter import Tk, ttk, messagebox, filedialog, Listbox, Button

class WavePlot():

    def __init__(self):
        self.metadata = self.looker()
        self.plotdata = None


    def looker(self):
        root = Tk()
        root.withdraw()
        filename = filedialog.askopenfilename(title = 'Select dataframe .txt format and space separation')
        print('Selection: %s' %(filename))
        df = pd.read_csv(filename, sep=' ')

        root.destroy()

        return df

    def select_folder(self, text):

        root = Tk()
        root.withdraw()
        foldername = filedialog.askdirectory(title=text)

        root.destroy()

        return foldername


    def welch_periodogram(self, path):

        waveform, sample_rate = torchaudio.load(path)

        resample_rate = 32000
        resampler = T.Resample(sample_rate, resample_rate, dtype = waveform.dtype)
        resampled_waveform = resampler(waveform)

        segment = int( 0.5*resample_rate )
        myhann = signal.get_window('hann', segment)

        myparams = dict(fs = resample_rate, nperseg = segment, window = myhann, noverlap = segment/2, scaling = 'spectrum', return_onesided=True)
        
        freq, ps = signal.welch( x = resampled_waveform, **myparams)
        ps = 2*ps

        myparams2 = dict(fs = resample_rate, nperseg = segment, window = myhann, noverlap = segment/2, scaling = 'density', return_onesided=True)

        freq, psd = signal.welch( x = resampled_waveform, **myparams2)
        psd = 2*psd

        dfreq = freq[1]

        normalized_psd = psd / np.linalg.norm(psd)
        
        return freq, normalized_psd


    def Mean_Power_Spectral_Density(self, color):

        folderpath = self.select_folder('Select wavs folder')

        stations = dict.fromkeys(self.metadata['station'].unique(), [])
        station_wav_freq = dict.fromkeys(self.metadata['station'].unique(), [])

        plot_data_list = []

        for i in range(len(self.metadata)):

            clear_output(wait = True)
            print('Working at:', folderpath)
            print(self.metadata.at[i, 'filename'])
            print(i+1, '/', len(self.metadata))

            wav_path = folderpath + '/' + self.metadata.at[i, 'filename']

            wav_freq, normalized_psd_dB = self.welch_periodogram(wav_path)
            
            stationFreq_aux = station_wav_freq[self.metadata.at[i, 'station']].copy()
            stationFreq_aux.append(wav_freq)
            station_wav_freq[self.metadata.at[i, 'station']] = stationFreq_aux

            station_aux = stations[self.metadata.at[i, 'station']].copy()
            station_aux.append(normalized_psd_dB)
            stations[self.metadata.at[i, 'station']] = station_aux

        figure, ax = plt.subplots(figsize = (13,8))
        

        for est in self.metadata['station'].unique():

            stations[est] = np.mean(stations[est], axis = 0)
            station_wav_freq[est] = np.mean(station_wav_freq[est], axis = 0)

            plot_data_list.append([est, station_wav_freq[est], stations[est].T, self.metadata.at[0, 'label']])

            plt.semilogx(station_wav_freq[est], stations[est].T , color = color[est], label = est, alpha=0.35)
        
        
        self.plotdata = pd.DataFrame(plot_data_list, columns = ['station', 'X [frequency]', 'Y [psd]', 'label'])

        title = 'Mean PSD of %s' %(self.metadata.at[0, 'label'])
        plt.xlabel('Frequency (Hz)'), plt.ylabel('Power spectral density ($dB/Hz)$') 
        plt.title(title, fontsize=12)
        plt.legend()



def PSD_plot(df, station, label):

    figure, ax = plt.subplots(figsize = (13,8))

    for i in range(len(df)):

        if(df.at[i, 'station'] in station):
            if(df.at[i, 'label'] in label):

                plt.semilogx(df.at[i, 'X [frequency]'], df.at[i, 'Y [psd]'].flatten(), label = [df.at[i, 'station'], df.at[i, 'label']], alpha = 0.45)
    
    
    stations_names = ' - '.join(station)
    label_names = ' - '.join(label)

    title = 'Mean psd of [ %s ] in [ %s ]' %(label_names, stations_names)
    plt.xlabel('Frequency (Hz)'), plt.ylabel('Power spectral density ($dB/Hz)$') 
    plt.title(title)
    plt.legend()

def norm_PSD_plot(df, station, label, alpha = 0.45, semilog = True, Title = True):

    figure, ax = plt.subplots(figsize = (13,8))

    for i in range(len(df)):

        if(df.at[i, 'station'] in station):
            if(df.at[i, 'label'] in label):


                if(semilog == True):

                    if(df.at[i, 'label'] == 'noise'):
                        plt.semilogx(df.at[i, 'X [frequency]'], df.at[i, 'Y [psd]'], label = [df.at[i, 'station'], df.at[i, 'label']], color = 'k')

                    else:
                        plt.semilogx(df.at[i, 'X [frequency]'], df.at[i, 'Y [psd]'], label = [df.at[i, 'station'], df.at[i, 'label']], alpha = alpha)



                elif(semilog == False):
                    
                    if(df.at[i, 'label'] == 'noise'):
                        plt.plot(df.at[i, 'X [frequency]'], df.at[i, 'Y [psd]'], label = [df.at[i, 'station'], df.at[i, 'label']], color = 'k')

                    else:
                        plt.plot(df.at[i, 'X [frequency]'], df.at[i, 'Y [psd]'], label = [df.at[i, 'station'], df.at[i, 'label']], alpha = alpha)


                
    
    
    stations_names = ' - '.join(station)
    label_names = ' - '.join(label)


    title = 'Normalized Mean PSD of [ %s ] in [ %s ]' %(label_names, stations_names)
    
    plt.xlabel('Frequency (Hz)'), plt.ylabel('Power spectral Density Unit Norm') 


    if( Title == True):
        plt.title(title)
    plt.legend()






