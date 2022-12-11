import numpy as np
import os
import pandas as pd
import umap.umap_ #pip install umap-learn
import umap.plot  #pip install umap-learn[plot]


from cepstrum import cepstrum
from IPython.display import Audio, display, clear_output, HTML
from tkinter import Tk, ttk, messagebox, filedialog, Listbox, Button

class Umaper:

    def __init__(self):
        self.dataframe = self.looker()


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

    def NPZ_generator(self):

        wavs_origin = self.select_folder('Select wavs origin folder')
        folder = wavs_origin.rsplit('/', 1)[-1]

        npz_destiny =self. select_folder('Select npz destiny folder')


        for index, row in self.dataframe.iterrows():

            npz_name = str(row['filename'].replace('.wav', '').replace('.mp3', ''))+'.npz'

            file_cepstrum = cepstrum(wavs_origin + '/' + row['filename'])

            np.savez(npz_destiny + '/' + npz_name, file_cepstrum)
            clear_output(wait = True)
            print('Working in %s folder' %(folder))
            print('%d / %d' %(index + 1 , len(os.listdir(wavs_origin))))
            print('Exporting %s in %s' %(npz_name, npz_destiny))




