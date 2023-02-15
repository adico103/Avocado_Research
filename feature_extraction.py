import os
import numpy as np
from data_loader import Sample_loader
import params_update

hpc=params_update.hpc

if hpc:
    saving_segmentation = 'segmentation//'
    saving_fourier = 'fourier//'
    data_folder =  'data_vnir'

else:
    data_folder =  params_update.data_folder

pca = []
l= []
for folder in os.listdir(data_folder):
    # try:
    if hpc:
        loader = Sample_loader(folder,data_folder,
                                saving_segmentation=saving_segmentation,
                                saving_fourier=saving_fourier)
        bands = loader.wavelength_range
        
    else:
        loader = Sample_loader(folder,data_folder,
                            saving_fourier ='fourier//' )
        bands = loader.wavelength_range[0::200]
    
    loader.segmentation(draw=True,load_file=True)
    loader.Process_by_band(bands=bands,fourier=False,save=False)
    loader.feature_extraction()


        

print('unfound files for folders:')
print(l)
