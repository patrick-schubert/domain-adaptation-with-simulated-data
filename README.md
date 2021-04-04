# Learning from Simulated Data in Highly Scarce Domains

Project from January - 2021 to March - 2021

This repository contains code for the **Learning from Simulated Data in Highly Scarce Domains** Project.

### Summary

This project aimed to propose a principled aproach to learn from simulated data and make inference with real data.

Dataset consist of samples from extremely rare events, Gravitational Lens. 

Objective: Estimate the Einstein Radius given the Multi-Spectral source.

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
#### Design Decisions

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
challenge1.adjust_distributions([99.6, 99.6, 99.6], [99.5, 99.5, 99.5], plot=True, normalize= True,
                                density = True, compare = real,  sanity = False, k = 4,bins = 100)
```

![hist](https://github.com/patrick-schubert/domain-adaptation-with-simulated-data/blob/main/imgs/hist.png)
![density](https://github.com/patrick-schubert/domain-adaptation-with-simulated-data/blob/main/imgs/density.png)

#### Pre-Processing Results

Finally we got fresh data that will make the learning process a lot easier for the learning process.


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

Many Deep Learning techniques were implemented in order to solve this regression problem. Some of them are novel and were developed exclusively for this problem.

Deep Learning techniques:

- Deterministic Shallower ResNet based model
- Bayesian Shallower ResNet based model
- Meta-Learning model
- Adversarial Training
- ShiftNet model **
- GAN based Domain Adaptation model **
- Continous Normalizing Flows based Domain Adaptation model **

** Custom and novel models designed by the course of work

A custom ResNet model was implemented in every Deep Learning technique applyied, a shallower one with 10 layers. 

#### Callbacks and Custom Tensorflow code

Custom callbacks for analysing in training results and a function to transform any deterministic Tensorflow 2 model in a probabilistic one were developed at the course the training process. 

```python3
class Probabilistic_Dispersion(tf.keras.callbacks.Callback):
    def __init__(self, monte_carlo):
        super().__init__()
        self.mc = monte_carlo
    
    def loss_prob_helper(self, real, pred_train):
        tmp = []
        for i in range(self.mc):
            tmp.append(sum(np.square(real - pred_train[:,i,:])) / real.shape[0])
        tmp = np.stack(tmp, axis=0)
        tmp_mean = tmp.mean(axis=0)[0]
        tmp_std = tmp.std(axis=0)[0]
        return tmp_mean, tmp_std

    def dispersion_prob_helper(self, real, pred_train):
        error = []
        for i in range(self.mc):
            error.append((real - pred_train[:,i,:]) / real)

        error = np.stack(error, axis=1)
        error_mean = error.mean(axis=1)
        error_std = error.std(axis=1)
        return error_mean, error_std
    
    def on_epoch_end(self, epoch, logs=None):
        
        fig, ax = plt.subplots(1,2, figsize = (10,5))
        
        indexes = np.random.randint(X_train.shape[0], size=batch_size)
        pred_train = [model.predict(X_train[indexes]) for _ in range(self.mc)]
        pred_train = np.stack(pred_train, axis=1)
        
        disp_mean, disp_std = self.dispersion_prob_helper(Y_train[indexes], pred_train)
        loss_mean, loss_std = self.loss_prob_helper(Y_train[indexes], pred_train)
        
        ax[0].scatter(Y_train[indexes], disp_mean)
        ax[0].set_title('Train')
        ax[0].legend([f" {loss_mean:0.3f} ± {loss_std:0.3f}"])
        ax[0].plot((0.5,2.5), (0,0), alpha = 0.3)
        ax[0].errorbar(Y_train[indexes], disp_mean, yerr = disp_std[:,0], linestyle = '', capsize = 5, alpha = 0.2, ecolor= 'b')
        
        
        pred_val = [model.predict(X_val) for _ in range(self.mc)]
        pred_val = np.stack(pred_val, axis=1)
        
        disp_mean, disp_std = self.dispersion_prob_helper(Y_val, pred_val)
        loss_mean, loss_std = self.loss_prob_helper(Y_val, pred_val)
        
        ax[1].scatter(Y_val, disp_mean)
        ax[1].set_title('Val')
        ax[1].legend([f" {loss_mean:0.3f} ± {loss_std:0.3f}"])
        ax[1].plot((0.5,2.5), (0,0), alpha = 0.3)
        ax[1].errorbar(Y_val, disp_mean, yerr = disp_std[:,0], linestyle = '', capsize = 5, alpha = 0.2, ecolor= 'b' )
        plt.show()
```

```python3
def replacer(model, to_be_replaced, replace, replace_args = {}, keep_args = []):
    def inputs_outputs(model):
        layers = [l for l in model.layers]
        outputs = []
        inputs = []
        for i in range(len(layers)):
            outputs.append(layers[i].output.name)
            if type(layers[i].input) == list:
                inputs.append([])
                for j in range(len(layers[i].input)):
                    inputs[i].extend([outputs.index(layers[i].input[j].name, 0, i)])

            else:
                inputs.append(outputs.index(layers[i].input.name))
        return inputs, outputs

    layers = [l for l in model.layers]
    inputs, outputs = inputs_outputs(model)


    inp_ = layers[0].input
    pointers = []
    pointers.append(inp_)
    for i in range(1, len(layers)):
        tmp = layers[i].__class__
        tmp_config = layers[i].get_config()

        if type(layers[i]) == to_be_replaced:
            
            tmp_config.update(replace_args)
            if keep_args != []:
                for remove in keep_args:
                    tmp_config.pop(remove)
            new_layer = replace(**tmp_config)
            x = new_layer(pointers[inputs[i]])
            pointers.append(x)
        else:
            if type(layers[i].input) == list:
                tmp_input = []
                for j in range(len(layers[i].input)):
                    tmp_input.append(pointers[inputs[i][j]])
            else:
                tmp_input = pointers[inputs[i]]
                
            new_layer = tmp(**tmp_config)
            x = new_layer(tmp_input)
            pointers.append(x)

    tmp_model = tf.keras.models.Model(inputs = inp_, outputs = x)
    
    return tmp_model

```

`Deterministic to Probabilistic in a function`

```python3
inp = ResNet10V2(X_val[0,:,:,:].shape)
y_hat = tf.keras.layers.Dense(1)(inp.layers[-2].output)

model = tf.keras.models.Model(inputs = inp.input, outputs = y_hat)
model.compile(optimizer = optimizer, loss = "mse")


kernel_divergence_fn = lambda q, p, _: tfd.kl_divergence(q, p) / tf.cast(X_train.shape[0], dtype="float32")
model = replacer(model, tf.keras.layers.Conv2D, 
                      tfp.layers.Convolution2DFlipout,
                      dict(kernel_divergence_fn = kernel_divergence_fn, bias_divergence_fn = kernel_divergence_fn), 
                      ['use_bias', 'bias_initializer', 'bias_regularizer', 'bias_constraint', 'kernel_initializer', 'kernel_regularizer', 'kernel_constraint'])
```



#### Model Realm

Model-Agnostic Meta-Learning (MAML) is a remarkable Deep Learning technique but sometimes their implementation and use comes a bit fuzzy. Below is a consise, fast and simple to use Tensorflow 2 MAML implementation designed by the course of the work.

```python3
#Run the model eagerly
class Maml(tf.keras.Model):
    def outter_data(self,x,y, k = 5, evenly = False):
        if evenly:
            space = np.ceil(len(y) / k).astype('int32')
            arg = np.argsort(y, axis=0)
            indexes = arg[::space].flatten()
        else:
            indexes = np.random.randint(0,y.shape[0], k)
        self.outter_x = tf.convert_to_tensor(x[indexes])
        self.outter_y = tf.convert_to_tensor(y[indexes])
    
    def copy_model(self, x):
        
        copied_model = Maml(self.inputs, self.outputs)
        copied_model.set_weights(self.get_weights())
        return copied_model

    def train_step(self, data):
        x, y = data
        
        with tf.GradientTape() as test_tape:
            with tf.GradientTape() as train_tape:
                y_pred = self(x, training=True)
                loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

            trainable_vars = self.trainable_variables
            gradients = train_tape.gradient(loss, trainable_vars)
        
            model_copy = self.copy_model(x)
            self.optimizer.apply_gradients(zip(gradients, model_copy.trainable_variables))

            test_y_pred = model_copy(self.outter_x, training=True)
            test_loss = self.compiled_loss(self.outter_y, test_y_pred, regularization_losses=self.losses)
            
        gradients = test_tape.gradient(test_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(y, y_pred)

        return {m.name: m.result() for m in self.metrics}
```

`Fast and simple to use`

```python3
inp = ResNet10V2(X_val[0,:,:,:].shape)
y_hat = tf.keras.layers.Dense(1)(inp.layers[-2].output)

model = Maml(inp.input, y_hat)
model.compile(optimizer = optimizer, loss = "mse", run_eagerly=True)
model.outter_data(X_val, Y_val, k = 5, evenly = True)
```

#### Novel Models

The models designed by the course of this work have many "bits and bytes". They are out of the scope os this presentation but preliminar code can be accessed here: [Lab](https://github.com/patrick-schubert/domain-adaptation-with-simulated-data/blob/main/Lab.ipynb)


### Results

After all workflow applyied at this task we achieved the desired goal of aproximate minimal 10% fractional error.

Plots below describe the fractional error between predictions and true values at Y axis and the entire Einstein Radius dataset range at X axis.

<table border="0">
  
<tr>
  <td >
  Before Pre-Processing Results
  </td>
  <td>
  After Pre-Processing Results
  </td>
</tr>
  
<tr>
    <td>
    <img src="https://github.com/patrick-schubert/domain-adaptation-with-simulated-data/blob/main/imgs/before_preprocesisng.png" width="100%" />
    </td>
    <td>
    <img src="https://github.com/patrick-schubert/domain-adaptation-with-simulated-data/blob/main/imgs/det_results.png", width="100%" />
    </td>
</tr>

<tr>
  <td >
  Probabilistc Results
  </td>
  <td>
  Best Model Results
  </td>
</tr>

<tr>
  <td>
  <img src="https://github.com/patrick-schubert/domain-adaptation-with-simulated-data/blob/main/imgs/prob_results.png" width="100%" />
  </td>
  <td>
  <img src="https://github.com/patrick-schubert/domain-adaptation-with-simulated-data/blob/main/imgs/best_results.png", width="100%" />
  </td>
</tr>


</table>

