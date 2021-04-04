import tensorflow as tf
import numpy as np
import copy

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
    
def ResNet10V2(input_shape = (224,224,3)):
    
    inp = tf.keras.layers.Input(shape=input_shape)

    x = conv_batchnorm_relu(inp, filters=64, kernel_size=7, strides=2)
    x = tf.keras.layers.MaxPool2D(pool_size = 3, strides =2)(x)
    x = resnet_block(x, filters=64//2, reps =2//2, strides=1)
    x = resnet_block(x, filters=128//2, reps =2//2, strides=2)
    x = resnet_block(x, filters=256//2, reps =2//2, strides=2)
    x = resnet_block(x, filters=512//2, reps =2//2, strides=2)
    x = tf.keras.layers.GlobalAvgPool2D()(x)

    output = tf.keras.layers.Dense(1000, activation ='softmax')(x)

    model = tf.keras.models.Model(inputs=inp, outputs=output)
    return model

def ResNet34V2(input_shape = (224,224,3)):
    
    inp = tf.keras.layers.Input(shape=input_shape)

    x = conv_batchnorm_relu(inp, filters=64, kernel_size=7, strides=2)
    x = tf.keras.layers.MaxPool2D(pool_size = 3, strides =2)(x)
    x = resnet_block(x, filters=64, reps =3, strides=1)
    x = resnet_block(x, filters=128, reps =4, strides=2)
    x = resnet_block(x, filters=256, reps =6, strides=2)
    x = resnet_block(x, filters=512, reps =3, strides=2)
    x = tf.keras.layers.GlobalAvgPool2D()(x)

    output = tf.keras.layers.Dense(1000, activation ='softmax')(x)

    model = tf.keras.models.Model(inputs=inp, outputs=output)
    return model


def conv_batchnorm_relu(x, filters, kernel_size, strides=1):
    
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    return x

def identity_block(tensor, filters):
    
    x = conv_batchnorm_relu(tensor, filters=filters, kernel_size=3, strides=1)
    x = conv_batchnorm_relu(x, filters=filters, kernel_size=3, strides=1)
    x = tf.keras.layers.BatchNormalization()(x)
    
    x = tf.keras.layers.Add()([tensor,x])    #skip connection
    x = tf.keras.layers.ReLU()(x)
    
    return x


def projection_block(tensor, filters, strides):
    
    #left stream
    x = conv_batchnorm_relu(tensor, filters=filters, kernel_size=3, strides=strides)
    x = conv_batchnorm_relu(x, filters=filters, kernel_size=3, strides=1)
    x = tf.keras.layers.BatchNormalization()(x)
    
    #right stream
    shortcut = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, strides=strides)(tensor)
    shortcut = tf.keras.layers.BatchNormalization()(shortcut)
    
    x = tf.keras.layers.Add()([shortcut,x])    #skip connection
    x = tf.keras.layers.ReLU()(x)
    
    return x

def resnet_block(x, filters, reps, strides):
    
    x = projection_block(x, filters, strides)
    for _ in range(reps-1):
        x = identity_block(x,filters)
        
    return x