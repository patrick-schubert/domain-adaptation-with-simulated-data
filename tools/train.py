import copy
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

def globalize(model_, batch_size_, x_train, y_train, x_val, y_val):
    global model, batch_size, X_train, Y_train, X_val, Y_val
    model, batch_size, X_train, Y_train, X_val, Y_val = model_, batch_size_, x_train, y_train, x_val, y_val

class Dispersion(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        
        fig, ax = plt.subplots(1,2, figsize = (10,5))
        
        indexes = np.random.randint(X_train.shape[0], size=batch_size)
        pred_train = model.predict(X_train[indexes])
        error_train = (Y_train[indexes] - pred_train) / Y_train[indexes]
        ax[0].scatter(Y_train[indexes], error_train)
        ax[0].set_title('Train')
        ax[0].legend(sum(np.square(Y_train[indexes] - pred_train)) / batch_size)
        ax[0].plot((0.5,2.5), (0,0))
        
        
        
        pred_val = model.predict(X_val)
        error_val = (Y_val - pred_val) / Y_val
        ax[1].scatter(Y_val, error_val)
        ax[1].legend(sum(np.square(Y_val - pred_val)) / pred_val.shape[0])
        ax[1].plot((0.5,2.5), (0,0))
        ax[1].set_title('Val')
        plt.show()
        
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

def sample_weight(X, granularity = 100):
    
    ### Density
    ep = 1e-7
    X_sorted = copy.deepcopy(X)
    X_sorted = np.sort(X_sorted, axis=0)
    bins = np.linspace(X_sorted.min() - ep, X_sorted.max() + ep, granularity, dtype = 'float32')
    events = np.zeros((granularity-1, 1), dtype = 'float32')
    for i in range(granularity - 1):
        for j in range(X_sorted.shape[0]):
            if X_sorted[j] < bins[i]:
                continue
            if (bins[i] <= X_sorted[j]) and (X_sorted[j] < bins[i+1]):
                events[i] += 1
    
    events /= events.sum()
    ###
    
    
    
    ### Sample_Weight
    sw = np.zeros_like(X, dtype = 'float32')
    for j in range(X.shape[0]):
        for i in range(granularity - 1):
            if (bins[i] <= X[j]) and (X[j] < bins[i+1]):
                sw[j] = events[i]
    
    max_val = sw.max()
    density = sw
    sw = max_val/sw
    ###
    return sw, density
    