import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torchaudio
import torchaudio.functional as functional
import torchaudio.transform as transform

from IPython.display import Audio, display, clear_output, HTML
from scipy import signal

class WavePlot():

    def __init__():
        self.metadata = self.looker()


    def looker(self):
        root = Tk()
        root.withdraw()
        filename = filedialog.askopenfilename(title = 'Select dataframe .txt format and space separation')

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

        waveformm, sample_rate = torchaudio.load(path)

        resample_rate = 32000
        resampler = transform.Resample(sample_rate, resample_rate, dtype = waveform.dtype)
        resampled_waveform = resampler(waveform)

        segment = int( 0.5*resample_rate )
        myhann = signal.get_window('hann', segment)

        myparams = dict(fs = resample_rate, nperseg = segment, window = myhann, noverlap = segment/2, scaling = 'spectrum', return_onesided=True)
        
        freq, ps = signal.welch( x = resampled_waveform, **myparams)
        ps = 2*ps

        myparams2 = dict(fs = resample_rate, nperseg = segment, window = myhann, noverlap = segment/2, scaling = 'density', return_onesided=True)

        freq, psd = signal.welch( x = resample_rate, **myparams2)
        psd = 2*psd

        dfreq = freq[1]

        normalized_psd = psd / np.linalg.norm(psd)
        normalized_psd_dB = 10*np.log10(normalized_psd)

        return freq, normalized_psd_dB


    def Mean_Power_Spectral_Density(self):

        folderpath = self.select_folder('Select wavs folder')

        mean_value = []

        for i in range(len(self.metadata)):
            clear_output(wait = True)
            print('Working at:', folderpath)
            print(self.metadata.at[i, 'filename'])
            print(i+1, '/', len(self.metadata))

            wav_path = folderpath + '/' + self.metadata.at[i, 'filename']

            wav_freq, normalized_psd_dB = welch_periodogram(wav_path)

            mean_value.append(normalized_psd_dB)

        normalized_psd_dB = np.array(normalized_psd_dB)
        normalized_psd_dB = np.mean(normalized_psd_dB, axis = 0)

        figure, ax = plt.subplots(figsize = (13,8))

        ax = plt.semilogx(wav_freq, mean_value.T, 'b', label = 'test')

        return ax 





