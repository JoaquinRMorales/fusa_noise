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
        normalized_psd_dB = 10*np.log10(normalized_psd)

        return freq, normalized_psd_dB


    def Mean_Power_Spectral_Density(self, color):

        folderpath = self.select_folder('Select wavs folder')

        stations = dict.fromkeys(self.metadata['station'].unique(), [])
        station_wav_freq = dict.fromkeys(self.metadata['station'].unique(), [])

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


            print(est)
            print('...')
            print(stations[est])
            print('-----------------------------------------------------')
            print(station_wav_freq[est])
            print('###########################################################')

            plt.semilogx(station_wav_freq[est], stations[est].T , color = color[est], label = est, alpha=0.4)
        
        
        title = 'PSD of %s' %(self.metadata.at[0, 'label'])
        plt.title(title, fontsize=12)
        plt.legend()






