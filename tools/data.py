from Utils.UtilsLens_Lu import loadLens_DataChallenge, LoadReal, New_LoadReal, New_loadchallenge2, lenspop_loader
import matplotlib.pyplot as plt
import tqdm
import cv2
import copy
from KDEpy import FFTKDE
import os 
import numpy as np
import astropy.io.fits as fits






class Dataset:
    def __init__(self, inputs, outputs, bands, sample_ids = None):
        self.inputs = inputs
        self.outputs = outputs
        self.sample_ids = sample_ids
        self.name = 'Dataset'
        self.bands = bands
    
    def shape(self):
        print(self.inputs.shape, self.outputs.shape, self.sample_ids.shape)
    
    def resize(self, dim, disable = False):
        tmp = []
        for i in tqdm.trange(self.inputs.shape[0], disable = disable):
            tmp.append(cv2.resize(self.inputs[i], dim))
        self.inputs = np.stack(tmp, axis=0).astype('float32')
    
    def normalizer(self,X):
        #min_v = X.min()
        #max_v = X.max()
        #return (X - min_v) / (max_v - min_v)
        mean_v = X.mean()
        std_v = X.std()
        return ( X - mean_v ) / std_v
    
    def dtype(self):
        print(self.name, 'Inputs',self.inputs.dtype)
        print(self.name, 'Outputs',self.outputs.dtype)
    
    def __plot_hist__(self, ax, data, x_lims = None, bins = 100 ,title = "Ponha um Título"):
        data = data.ravel()
        bins = np.linspace(data.min(), data.max(), bins)
        ax.hist(data, bins = bins)
        ax.set_title(title)
        if x_lims != None:
            ax.set_xlim(x_lims[0], x_lims[1])
    
    def plot_distributions(self, x_lims = None, inp = None, bins = 100):
        fig, ax = plt.subplots(1,len(self.bands) + 1, figsize = (30,5))
        
        if x_lims == None:
            x_lims = [x_lims]*len(self.bands)
        
        if type(inp) == type(np.array(0)):
            inputs = inp
        else:
            inputs = self.inputs
    
        for i in range(len(self.bands)):
            self.__plot_hist__(ax[i], inputs[:,:,:,i], x_lims[i], bins,
              (f"{self.name} {self.bands[i]} | Max: {inputs[:,:,:,i].max():.3e}\n"\
               f"Min: {inputs[:,:,:,i].min():.3e}"))
        self.__plot_hist__(ax[-1], self.outputs, 
              title = (f"{self.name} 'Outputs' | Max: {self.outputs.max():.3e}\n"\
               f"Min: {self.outputs.min():.3e}"))
        fig.show()
        
    def adjust_distributions(self, bands_max_percentiles, bands_min_percentiles, plot=False, normalize = False, density = False, compare = None ,sanity = False, k = 0, bins = 100, ):

        if (len(self.bands) != len(bands_max_percentiles)) or (len(self.bands) != len(bands_min_percentiles)):
            print("Numero de bands diferete do fornecido nos percentiles")
        else:
            percentile_list = []
            inp = []
            for band in range(len(self.bands)):
                band_max = self.inputs[:,:,:,band].max()
                max_percentile = np.percentile(self.inputs[:,:,:,band], bands_max_percentiles[band])
                band_min = self.inputs[:,:,:,band].min()
                min_percentile = -(np.percentile(-self.inputs[:,:,:,band], bands_min_percentiles[band]))
                prefix = f"{self.name} | Band {self.bands[band]} "
                print(f"\n{prefix}|   Max: {band_max:0.3e} |   Min: {band_min:0.3e}")
                prefix = " "*(len(prefix))
                print(f"{prefix}| P_max: {max_percentile:0.3e} | P_min: {min_percentile:0.3e}")
                if k != 0:
                    print(f"\nTopK_max: {(self.__topk__(self.inputs[:,:,:,band], k, kind = 'max'))}\nTopK_min: {(self.__topk__(self.inputs[:,:,:,band], k, kind = 'min'))}")

                percentile_list.append([min_percentile, max_percentile])
                tmp = np.clip(self.inputs[:,:,:,band], min_percentile, max_percentile)
                
                if normalize:
                    #tmp = (tmp - min_percentile)/(max_percentile - min_percentile)
                    tmp = self.normalizer(tmp)
                
                inp.append(tmp)
                    
            inp = np.stack(inp, axis=-1)
            
            if normalize:
                self.inputs = inp

            if plot and not(normalize):
                self.plot_distributions(percentile_list, inp, bins = bins)
                
            if plot and normalize:
                self.plot_distributions()
                
            if density:
                if compare != None:
                    self.density(compare)
                else:
                    self.density()
                
            if sanity:
                print("\nAfter Preprocessing:\n")
                for band in range(len(self.bands)):
                    band_max = inp[:,:,:,band].max()
                    band_min = inp[:,:,:,band].min()
                    prefix = f"{self.name} | Band {self.bands[band]} "
                    print(f"{prefix}|   Max: {band_max:0.3e} |   Min: {band_min:0.3e}")
                for band in range(len(self.bands)):
                    print(f"\nTopK_max: {(self.__topk__(inp[:,:,:,band], k, kind = 'max'))}\nTopK_min: {(self.__topk__(inp[:,:,:,band], k, kind = 'min'))}")
                    
                self.plot_samples(inp, normalize = not(normalize))
                    
                    
    def plot_samples(self, inp = None, normalize = False):
        
        if type(inp) == type(np.array(0)):
            inputs = inp
        else:
            inputs = self.inputs
            
        if self.name == "Challenge2":
            inputs = np.concatenate([inputs, inputs[:,:,:,0:1]], axis=-1)
            
        if normalize:
            for i in range(inputs.shape[-1]):
                inputs[:,:,:,i] = self.normalizer(inputs[:,:,:,i])
        
        row, col = 6,5
        fig, ax = plt.subplots(row,col, figsize = (30,25))
        idx = 0
        fig.subplots_adjust(wspace=0.2, hspace=0.2)
        for i in range(row):
            for j in range(col):
                try:
                    
                   
                    sample = np.stack([inputs[idx,:,:,0], inputs[idx,:,:,1], inputs[idx,:,:,2]], axis=-1)
                    ax[i,j].imshow(sample)
                    idx+=1
                except:
                    ax[i,j].axis('off')
        plt.show()
        
    def __topk__(self, X, k, kind = 'max'):
        X = X.ravel()
        if kind == 'max':
            ind = np.argpartition(X, -k)[-k:]
        else:
            ind = np.argpartition(-X, -k)[-k:]
        return X[ind[np.argsort(X[ind])]]
    
    def density(self, compare = None):


        fig, ax = plt.subplots(1,len(self.bands) + 1, figsize = (30,5))
        eval_list = []
        ep = 1e-10
        for i in range(len(self.bands)):

            kde = FFTKDE('gaussian', bw = 0.13)
            kde.fit(self.inputs[:,:,:,i].ravel())

            if compare != None:
                min_v, max_v = self.domain(self.inputs[:,:,:,i], compare.inputs[:,:,:,i])
                grid = np.linspace(min_v - ep, 
                               max_v + ep, 100)
            else:
                grid = np.linspace(self.inputs[:,:,:,i].min() - ep, 
                               self.inputs[:,:,:,i].max() + ep, 100)
            evaluation = kde.evaluate(grid)
            ax[i].plot(grid, evaluation, label = self.name)
            ax[i].set_title(f"{self.name} {self.bands[i]}")
            eval_list.append(evaluation)

        kde = FFTKDE('gaussian', bw = 0.13)
        kde.fit(self.outputs)
        if compare != None:
            min_v, max_v = self.domain(self.outputs, compare.outputs)
            grid = np.linspace(min_v - ep, 
                           max_v + ep, 100)
        else:
            grid = np.linspace(self.outputs.min() - ep, 
                           self.outputs.max() + ep, 100)

        evaluation = kde.evaluate(grid)
        ax[-1].plot(grid, evaluation, label = self.name)
        ax[-1].set_title(f"{self.name} Outputs")
        eval_list.append(evaluation)


        if compare != None:
            for i in range(len(self.bands)):

                kde = FFTKDE('gaussian', bw = 0.13)
                kde.fit(compare.inputs[:,:,:,i].ravel())
                #if compare != None:
                min_v, max_v = self.domain(self.inputs[:,:,:,i], compare.inputs[:,:,:,i])
                grid = np.linspace(min_v - ep, 
                               max_v + ep, 100)

                ax[i].plot(grid, kde.evaluate(grid), label = compare.name)
                ax[i].plot(grid, kde.evaluate(grid) - eval_list[i], label= "Difference")
                ax[i].set_title(f"{self.name} {self.bands[i]} | Compare: {compare.name}")
                ax[i].plot([self.inputs[:,:,:,i].min(), self.inputs[:,:,:,i].max()], [0.0, 0.0], linestyle = '--', alpha = 0.3)

            kde = FFTKDE('gaussian', bw = 0.13)
            kde.fit(compare.outputs)
            #if compare != None:
            min_v, max_v = self.domain(self.outputs, compare.outputs)
            grid = np.linspace(min_v - ep, 
                           max_v + ep, 100)

            ax[-1].plot(grid, kde.evaluate(grid), label = compare.name)
            ax[-1].plot(grid, kde.evaluate(grid) - eval_list[-1], label= "Difference")
            ax[-1].set_title(f"{self.name} Outputs | Compare: {compare.name}")
            ax[-1].plot([self.outputs.min(), self.outputs.max()], [0.0, 0.0], linestyle = '--', alpha = 0.3)


        plt.legend()
        plt.show()
        
    def domain(self,X1,X2):
        min_v = X1.min()
        max_v = X1.max()
        if min_v > X2.min():
            min_v = X2.min()
        if max_v < X2.max():
            max_v = X2.max()

        return min_v, max_v
    
    def pad(self, X, reverse = False):
        shape = X.shape[:3]
        zeros = np.zeros((*shape, 1), dtype="float32")
        if reverse == False:
            return np.concatenate([X, zeros], axis=-1)
        else:
            return np.concatenate([zeros, X], axis=-1)
        
    def post_processing(self, sanity=False):
        if sanity:
            print("BEFORE:")
            for i in range(len(self.bands)):
                print(f"{self.name} | Mean: {self.inputs[:,:,:,i].mean()} | STD: {self.inputs[:,:,:,i].std()}")
        for i in range(len(self.bands)):
            self.inputs[:,:,:,i] = self.normalizer(self.inputs[:,:,:,i])

        if sanity:
            print("AFTER:")
            for i in range(len(self.bands)):
                print(f"{self.name} | Mean: {self.inputs[:,:,:,i].mean()} | STD: {self.inputs[:,:,:,i].std()}")
            print()

#['i','r','g']
class LensPop(Dataset):
    def __init__(self):
        
        DATA_DIR = '/home/dados229/Data/LensPop/'
        CATALOG = 'ground_truth_catalog.json'
        BANDS = ['i','r','g']
        N_PIX = 20
        ER_MIN = 0.0
        ER_MAX = 2.5
        SAVED = False
        
        inputs, outputs, sample_ids = lenspop_loader(DATA_DIR, CATALOG, BANDS,
                                                         N_PIX, ER_MIN, ER_MAX, SAVED)
        
        self.inputs = inputs["images"]
        self.outputs = outputs["einstein_radius"]
        self.sample_ids = sample_ids
        self.name = 'LensPop'
        self.bands = BANDS
        

#Bands = ['R', 'I', 'G'] + ['U']
class Challenge1(Dataset):
    def __init__(self):
        DirB = '/home/dados229/Data/Challenge1/'
        dirLenPop = '/home/dados229/Data/Challenge1/'
        dirLenPopAll = dirLenPop

        DirSav = '/home/cenpes240/P/deep-kernel-transfer/Challenge1/'
        DirNPY = DirSav
        dirCatal = '/home/dados229/Data/Challenge1/'

        NN = None
        corte = 76#76
        NSAMPLES = 20000
        n_pix = 100
        ER_MAX= 2.5
        ER_MIN = 0.0
        saved = False
        Bands = ['I', 'R', 'G']
        inputs, outputs, sample_ids = loadLens_DataChallenge(DirB, DirSav, DirNPY, dirLenPopAll, 
                                                             dirCatal, NSAMPLES, NN, n_pix, 
                                                             ER_MAX, ER_MIN, saved, Bands,corte=corte)
        
        
        self.inputs = inputs["images"]
        self.inputs = np.where(self.inputs != 100., self.inputs , 0.0)
        self.outputs = outputs["einstein_radius"]
        self.sample_ids = sample_ids
        self.name = 'Challenge1'
        self.bands = Bands
        
#Bands = ['i', 'r', 'g'] + ['z', 'Y']
class Real(Dataset):
    def __init__(self):
        OUTS = ['einstein_radius']

        DirBase = '/home/dados229/Data/Real_Lens/'
        DirImg = DirBase 
        catt = DirBase + 'lenses_DES.txt'
        catt2 = DirBase + 'lenses_DES_standarized.txt'

        Bands = ['i','r','g']
        inputs, outputs, sample_ids = New_LoadReal(catt,catt2,DirImg,OUTS,Bands)
        
        self.inputs = inputs["images"]
        self.outputs = outputs["einstein_radius"]
        self.sample_ids = sample_ids
        self.name = 'Real'
        self.bands = Bands
        
        #BETA
        self.outputs = np.delete(self.outputs, 4).reshape(-1,1)
        self.inputs = np.delete(self.inputs, 4, axis=0)
        self.sample_ids = np.delete(self.sample_ids, 4 , axis=0)


#Bands = ['H', 'J', 'VIS', 'Y']
class Challenge2(Dataset):
    def __init__(self, BANDS = ['Y']):
        
        DATA_DIR = '/home/dados229/Data/Challenge2/'
        CATALOG = 'image_catalog2.0train_corrigido.csv'
        
        N_PIX = 100
        ER_MAX= 2.5
        ER_MIN = 0.0
        SAVED = False
        #BANDS = ['Y']
        PARALLEL = 'Process' #Thread_Future, Thread, Process, False
        
        inputs, outputs, sample_ids = New_loadchallenge2(DATA_DIR, CATALOG, BANDS,
                                                         N_PIX, ER_MIN, ER_MAX, SAVED,
                                                         PARALLEL)
        
        self.inputs = inputs["images"]
        self.outputs = outputs["einstein_radius"]
        self.sample_ids = sample_ids
        self.name = 'Challenge2'
        self.bands = BANDS
        
    def concatenate(self, X):
        if self.inputs.shape != X.inputs.shape:
            X.resize(self.inputs.shape[1:3], disable = True)
            
        #self.inputs = np.concatenate( [self.inputs, X.inputs.reshape(*X.inputs.shape,1)] , axis=-1)
        if self.inputs.ndim == 3:
            self.inputs = np.concatenate( [self.inputs.reshape(*self.inputs.shape,1), X.inputs.reshape(*X.inputs.shape,1)] , axis=-1)
        else:
            self.inputs = np.concatenate( [self.inputs, X.inputs.reshape(*X.inputs.shape,1)] , axis=-1)
        self.bands.extend(X.bands)
