import math
import numpy as np
import os
import pandas as pd
import sys
import torch
import torchaudio

from IPython.display import Audio, display, clear_output, HTML
from tkinter import Tk, ttk, messagebox, filedialog, Listbox, Button


class Wavs:

    def __init__(self):
        # self.thrshold = thrshold
        self.metadata = self.looker()

    def looker(self):
        root = Tk()
        root.withdraw()
        filename = filedialog.askopenfilename(title = 'Select dataframe .txt format and space separation')

        df = pd.read_csv(filename, sep=' ')

        root.destroy()

        return df
        
    def export_data(self, dataframe, filename, targetfolder):

        foldername = targetfolder + '/' + filename + '.txt'


        try:
            dataframe.to_csv(foldername, sep = ' ', mode = 'a', index = 0 )
            print('Data exported at:', foldername)

        except :
            print("Oops!", sys.exc_info()[0], 'occurred,')


    def audio_power(self, path, filename):
        
        
        audio_path = path + '/' + filename

        waveform, sample_rate = torchaudio.load(audio_path)
        num_channles, num_frames = waveform.shape

        if waveform.numpy().shape[1] != 0:
            
            waveform_fft = np.fft.fft(waveform.numpy())/math.sqrt(waveform.numpy().shape[1])
            parseval_transform = np.sum(np.abs(waveform_fft)**2)


            duration = torch.arange(0, num_frames) / sample_rate
            duration = duration.numpy()[-1]

            return duration, parseval_transform

        else:
            print('Waveform error')
            return None, None



    def classifier(self, threshold):

        foldername = self.select_folder('Select wavs folders for classification')
        folder = foldername.rsplit('/', 1)[-1]

        targetfolder = self.select_folder('Select dataframe destiny folder')

        columns = ['filename', 'original wav', 'label', 'station', 'duration', 'power']
        df = pd.DataFrame(columns=columns)

        filtered_audios = 0

        for i in range(len(os.listdir(foldername))):
            clear_output(wait=True)
            print('%s folder is being classified' %(folder))
            print(i+1, '/' ,len(os.listdir(foldername)))
            print('Ignored audios [threshold = %f sec]: %d' %(threshold, filtered_audios))
            row = []

            for x in range(len(self.metadata)):
                    
                if(str(self.metadata.at[x, 'filename'].replace('.wav', '')).replace('.mp3', '') in os.listdir(foldername)[i]):


                    row_filename = os.listdir(foldername)[i]
                    row_originalwav = str(self.metadata.at[x, 'filename'].replace('.wav', '')).replace('.mp3', '')
                    row_station = self.metadata.at[i, 'station']
                    row_duration, row_power = self.audio_power(foldername, row_filename)

                    row = [ row_filename, row_originalwav, folder, row_station, row_duration, row_power]
                    if(row_duration >= threshold):
                        df.loc[len(df)] = row
                    else:
                        filtered_audios += 1

                    break
        


        self.export_data(df, folder, targetfolder)
 


    def select_folder(self, text):

        root = Tk()
        root.withdraw()
        foldername = filedialog.askdirectory(title=text)

        root.destroy()

        return foldername
                




class File:

    def __init__(self):
        print('Hi!')


#How to use
#First, read original metadata for classification purpose
#   Create Wavs Object, like wavs = Wavs() -> Trigger looker function
#       -> This will ask for a file with txt format and space separated values
#       -> Metadata will be stored in class atribute "metadata"

#Second, with the original data, we need to create dataframes for noise and events values, for this, we are going to search in metadata the wavs characteristics
#   Use object function classifier
#          -> function argument is threshold of audio duration
#       -> This function creates a dataframe with wavs filename, origin, label, station, audio duration and power

if __name__ == "__main__":
    wavs = Wavs()
    print(wavs.metadata)
    wavs.classifier(0.5)