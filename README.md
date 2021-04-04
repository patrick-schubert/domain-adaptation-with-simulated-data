# Learning from Simulated Data in Highly Scarce Domains

Project from January - 2021 to March - 2021

This repository contains code for the **Learning from Simulated Data in Highly Scarce Domains** Project.

### Summary

This project aimed to propose a principled aproach to learn from simulated data and make inference with real data.

Dataset consist of samples from extremely rare events, Gravitational Lens.

<img src="https://github.com/patrick-schubert/domain-adaptation-with-simulated-data/blob/main/imgs/ilustration.jpg" width="50%" />


### Dataset

**Training set** is an aggregate from 3 different simulations that then comes from different distributions and are highly noisy and unbalanced.

**Test set** consists of aproximatetly only 30 samples from real events taken from different telescopes.

<table border="0">
<tr>
  <td >
  Simulated
  </td>
  <td>
  Real
  </td>
</tr>
<tr>
    <td>
    <img src="https://github.com/patrick-schubert/domain-adaptation-with-simulated-data/blob/main/imgs/samples_pre2.png" width="100%" />
    </td>
    <td>
    <img src="https://github.com/patrick-schubert/domain-adaptation-with-simulated-data/blob/main/imgs/samples_pre.png", width="100%" />
    </td>
</tr>
</table>


Some simulated datasets are huge and claimed the need for a multiprocessing loading strategy. 

Data went from 30 min loading time to 1 min, aprox 30x faster.

```python3
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
```

```python3
slices = []
size = catalog.shape[0]
slice_width = size // cpu_count()

for i in range(0, size, slice_width):
    if i + slice_width > size:
        slices.append((i, size))
    else:
        slices.append((i, i + slice_width))

ID_LIST = catalog['ID'].values.tolist()

with Pool(cpu_count()) as p:
    pool_outputs = list(tqdm.tqdm_notebook(p.map(parallel_loader, slices),
                       total = len(slices), desc= 'Challenge2 Data ' + channels[0]))

images = np.concatenate(pool_outputs, axis=0)
```
#### Data Structures

Datasets were implemented with OOP to grant cohesian between data sources and give an easy time for everyone working around with them, making them iterate through the data analysis step faster.

Dataset class methods:
- shape
- resize
- normalizer
- dtype
- __plot_hist__
- plot_distributions
- adjust_distributions
- plot_samples
- __topk__
- density
- domain
- pad
- post_processing

Specific datasets inherits from the abstract class "Dataset"

```python3
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
       
```



### Pre-Processing

#### Bands
 
Most datasets in this project comes from 3 feature bands (g, r, i) that captures most of the visible spectrum available, but our bigger simulated dataset comes from another physical process that defines their domain of distribution from the VIS band.

Some **signal processing**  steps were taken to ensure that every sample comes from the closest distribution possible.

This step was crucial to make things work as expected as their absense made the learning process worse.

<table border="0">
<tr>
  <td >
  Datasets 1 - 2 and Real
  </td>
  <td>
  Dataset 3
  </td>
</tr>
<tr>
    <td>
    <img src="https://github.com/patrick-schubert/domain-adaptation-with-simulated-data/blob/main/imgs/spectral-gri.jpg" width="100%" />
    </td>
    <td>
    <img src="https://github.com/patrick-schubert/domain-adaptation-with-simulated-data/blob/main/imgs/spectral-vis.jpg", width="70%" />
    </td>
</tr>
</table>

#### Distribution Shifts

Pre-Processing decisions were taken as to minimize the difference between the density of simulated datasets and the real one. This process ensures that sample data can only "wiggle" around inside the real distribution, making the expected simulated sample closer to the real. 

```python3
challenge1.adjust_distributions([99.6, 99.6, 99.6], [99.5, 99.5, 99.5], plot=True, normalize= True,density = True, 
                                                                        compare = real,  sanity = False, k = 4,bins = 100)
```

![hist](https://github.com/patrick-schubert/domain-adaptation-with-simulated-data/blob/main/imgs/hist.png)
![density](https://github.com/patrick-schubert/domain-adaptation-with-simulated-data/blob/main/imgs/density.png)

#### Pre-Processing Results

Finally we got fresh data that will make the learning process a lot easier for the model.


<table border="0">
  
<tr>
  <td >
  Before Pre-Processing
  </td>
  <td>
  After Pre-Processing
  </td>
</tr>
  
<tr>
    <td>
    <img src="https://github.com/patrick-schubert/domain-adaptation-with-simulated-data/blob/main/imgs/samples_pre.png" width="100%" />
    </td>
    <td>
    <img src="https://github.com/patrick-schubert/domain-adaptation-with-simulated-data/blob/main/imgs/samples_post1.png", width="100%" />
    </td>
</tr>
  
<tr>
  <td>
  <img src="https://github.com/patrick-schubert/domain-adaptation-with-simulated-data/blob/main/imgs/samples_pre2.png" width="100%" />
  </td>
  <td>
  <img src="https://github.com/patrick-schubert/domain-adaptation-with-simulated-data/blob/main/imgs/samples_post2.png", width="100%" />
  </td>
</tr>

<tr>
    <td>
    <img src="https://github.com/patrick-schubert/domain-adaptation-with-simulated-data/blob/main/imgs/samples_pre3.png" width="100%" />
    </td>
    <td>
    <img src="https://github.com/patrick-schubert/domain-adaptation-with-simulated-data/blob/main/imgs/samples_post3.png", width="100%" />
    </td>
</tr>

</table>


### Learning Process



