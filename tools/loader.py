import tensorflow as tf
import numpy as np

from astropy.nddata import Cutout2D
from astropy.io import fits
from astropy.io import ascii

import cv2
import bisect
import os.path
import math as mt
import pickle as pkl
import getpass, datetime
import os, platform

from time import time
import tqdm
import json
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor,as_completed

import warnings
warnings.filterwarnings("ignore")
import csv
import pandas as pd
import pandas

def parallel_loader(slices_tuple):
    
    tmp_catalog = ID_LIST[slices_tuple[0]:slices_tuple[1]]
    images = None

    for iid,cid in enumerate(tmp_catalog):
        for ich,ch in enumerate(CHANNELS_P):
            image_file = os.path.join(DATA_DIR_P,
                                      'Train',
                                      'Public',
                                      'EUC_' + ch,
                                      'imageEUC_{}-{}.fits'.format(ch,cid))

            if os.path.isfile(image_file):
                image_data = fits.getdata(image_file, ext=0).astype("float32")
                if images is None:
                    images = np.zeros((len(tmp_catalog),*image_data.shape,len(CHANNELS_P)),
                                     dtype = 'float32')

                images[iid,:,:,ich] = image_data

    return images

def New_loadchallenge2(DATA_DIR, CATALOG, CHANNELS, N_PIX, ER_MIN, ER_MAX, SAVED, PARALLEL):
    
    data_dir = DATA_DIR
    catalog_name = CATALOG
    global ID_LIST
    global CHANNELS_P
    global DATA_DIR_P
    
    CHANNELS_P = CHANNELS
    DATA_DIR_P = DATA_DIR

    """ Load catalog before images """
    catalog = pd.read_csv(os.path.join(data_dir, catalog_name), header = 0) # 28 for old catalog

    """ Now load images using catalog's IDs """
    channels = CHANNELS
    nsamples = len(catalog['ID'])
    idxs2keep = []


    """ Try to load numpy file with images """
    if os.path.isfile(os.path.join(data_dir,'images_hjy.npy')) and SAVED:  
        images = np.load(os.path.join(data_dir,'images_hjy.npy'))
        idxs2keep = list(np.load(os.path.join(data_dir,'idxs2keep.npy')))
    else:
        images = None
        for iid,cid in enumerate(catalog['ID']): 
            for ich,ch in enumerate(channels):
                image_file = os.path.join(data_dir,
                                          'Train',
                                          'Public',
                                          'EUC_' + ch,
                                          'imageEUC_{}-{}.fits'.format(ch,cid))
        
                if os.path.isfile(image_file):
                    idxs2keep.append(iid)
                    break
                else:
                    break

        r = lambda A: np.sqrt(np.float(A) / np.pi)*206265
        catalog = catalog.iloc[idxs2keep]
        catalog['ein_area'] = catalog['ein_area'].apply(r)
        is_lens = np.where((catalog['n_source_im'] > 0) & (catalog['mag_eff'] > 1.6) & (catalog['n_pix_source'] > N_PIX) & (catalog['ein_area'] > ER_MIN) & (catalog['ein_area'] < ER_MAX)  )[0]
        catalog = catalog.iloc[is_lens]

        if PARALLEL == 'Thread_Future': 
            slices = []
            size = catalog.shape[0]
            slice_width = size // cpu_count()

            for i in range(0, size, slice_width):
                if i + slice_width > size:
                    slices.append((i, size))
                else:
                    slices.append((i, i + slice_width))
        
            ID_LIST = catalog['ID'].values.tolist()

            def parallel_loader(slices_tuple, pbar):
                tmp_catalog = ID_LIST[slices_tuple[0]:slices_tuple[1]]
                images = None

                for iid,cid in enumerate(tmp_catalog): 
                    for ich,ch in enumerate(channels):
                        image_file = os.path.join(data_dir,
                                                  'Train',
                                                  'Public',
                                                  'EUC_' + ch,
                                                  'imageEUC_{}-{}.fits'.format(ch,cid))

                        if os.path.isfile(image_file):

                            image_data = fits.getdata(image_file, ext=0).astype("float32")
                            if images is None:
                                images = np.zeros((len(tmp_catalog),*image_data.shape,len(channels)),
                                                 dtype = 'float32')

                            images[iid,:,:,ich] = image_data
                    pbar.update(1)
                return images
            with tqdm.tqdm_notebook(total=catalog.shape[0]) as pbar:
                with ThreadPoolExecutor(max_workers = cpu_count()) as p:
                    pool_outputs = [p.submit(parallel_loader, slices_tuple, pbar) for slices_tuple in slices]

            pool_outputs = list(map(lambda x : x.result(), pool_outputs))
            images = np.concatenate(pool_outputs, axis=0)
            
        elif PARALLEL == 'Thread': 
            from threading import Lock
            lock = Lock()
           
            slices = []
            size = catalog.shape[0]
            slice_width = size // cpu_count()

            for i in range(0, size, slice_width):
                if i + slice_width > size:
                    slices.append((i, size))
                else:
                    slices.append((i, i + slice_width))
        
            ID_LIST = catalog['ID'].values.tolist()

            def parallel_loader(slices_tuple):

                tmp_catalog = ID_LIST[slices_tuple[0]:slices_tuple[1]]
                images = None
                for iid,cid in enumerate(tmp_catalog):
                    for ich,ch in enumerate(channels):
                        image_file = os.path.join(data_dir,
                                                  'Train',
                                                  'Public',
                                                  'EUC_' + ch,
                                                  'imageEUC_{}-{}.fits'.format(ch,cid))

                        if os.path.isfile(image_file):
                            image_data = fits.getdata(image_file, ext=0).astype("float32")
                            if images is None:
                                images = np.zeros((len(tmp_catalog),*image_data.shape,len(channels)),
                                                 dtype = 'float32')
                            images[iid,:,:,ich] = image_data
                    with lock:
                        pbar.update(1)
                return images

            with tqdm.tqdm_notebook(total=catalog.shape[0]) as pbar:
                with ThreadPool(cpu_count()) as p:
                    pool_outputs = p.map(parallel_loader, slices)

            images = np.concatenate(pool_outputs, axis=0)
        
        
        elif PARALLEL == 'Process': 
           
            slices = []
            size = catalog.shape[0]
            slice_width = size // cpu_count()

            for i in range(0, size, slice_width):
                if i + slice_width > size:
                    slices.append((i, size))
                else:
                    slices.append((i, i + slice_width))
            
            ID_LIST = catalog['ID'].values.tolist()
            
            

            from Utils.UtilsLens_Lu import parallel_loader
            
            with Pool(cpu_count()) as p:
                pool_outputs = list(tqdm.tqdm_notebook(p.map(parallel_loader, slices),
                                   total = len(slices), desc= 'Challenge2 Data ' + channels[0]))

            images = np.concatenate(pool_outputs, axis=0)
            
        else:
            images = None
            for iid,cid in enumerate(tqdm.tqdm_notebook(catalog['ID'], desc='Challenge2 Data')):
                for ich,ch in enumerate(channels):
                    image_file = os.path.join(data_dir,
                                              'Train',
                                              'Public',
                                              'EUC_' + ch,
                                              'imageEUC_{}-{}.fits'.format(ch,cid))

                    if os.path.isfile(image_file):
                        image_data = fits.getdata(image_file, ext=0).astype("float32")
                        if images is None:
                            images = np.zeros((catalog.shape[0],*image_data.shape,len(channels)),
                                             dtype = 'float32')

                        images[iid,:,:,ich] = image_data 
                        
        inputs = {'images': images}
        outputs = {'einstein_radius': np.array([catalog['ein_area'].values]).T}
        sample_ids = (np.array(idxs2keep)[is_lens]).reshape(-1,1)

        return inputs, outputs, sample_ids
