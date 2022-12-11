import sys
import os

modules_route = str(os.getcwd()).replace("\\", "/")+"/Modules/"
sys.path.insert(0, modules_route)

import waver as wv


wavs = wv.Wavs()
print(wavs.metadata)
wavs.export_data('noise')