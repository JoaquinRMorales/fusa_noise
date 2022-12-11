import umap.umap_
import umap.plot

from cepstrum import cepstrum
from tkinter import Tk, ttk, messagebox, filedialog, Listbox, Button

class Umaper:

    def __init__(self):
        self.dataframe = self.looker()


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

    def NPZ_generation():

        wavs_origin = select_folder('Select wavs origin folder')
        folder = wavs_origin.rsplit('/', 1)[-1]

        npz_destiny = select_folder('Select npz destiny folder')

        for index, row in self.dataframe.iterrows():

            npz_name = str(row['filename'].replace('.wav', '').replace('.mp3', ''))+'.npz'

            print(npz_name)




