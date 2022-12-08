import os
import pandas as pd
import sys

from tkinter import Tk, ttk, messagebox, filedialog, Listbox, Button


class Wavs:

    def __init__(self, thrshold):
        self.thrshold = thrshold
        self.metadata = self.looker()
        self.files = []

    def add_file(self, file):
        if file.duration >= self.thrshold:
            self.files.append(file)

    def looker(self):
        root = Tk()
        root.withdraw()
        filename = filedialog.askopenfilename()

        df = pd.read_csv(filename, sep=' ')

        root.destroy()

        return df
        
    def export_data(self):

        root = Tk()
        root.withdraw()
        foldername = filedialog.askdirectory()
        foldername = foldername + '/wavs_metadata.txt'

        try:
            self.metadata.to_csv(foldername, sep = ' ', mode = 'a')
            print('Data exported!')

        except :
            print("Oops!", sys.exc_info()[0], 'occurred,')




class File:

    def __init__(self):
        print('Hi!')


if __name__ == "__main__":
    wavs = Wavs(0.5)
    print(wavs.metadata)
    wavs.export_data()


