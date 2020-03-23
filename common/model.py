import os
import pandas as pd
import tensorflow as tf

from data_manager import *
from log_manager  import *
# from layer_v_2 import *

class Model:
    def __init__(self, log_dir, name, lr, data_manager, log_manager):
        self.name = name
        self.dm = data_manager
        self.lm = log_manager
        self.lr = lr
        self.x = None 
        self.y = None
        self.layers = []
        self.activations = []
        
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True 
        self.sess = tf.compat.v1.Session(config=config)
        
        self.ckptdir = os.path.join(log_dir, self.name+'/')
        self.lm.add_line('Check point location: {}'.format(self.ckptdir))
        if os.path.exists(self.ckptdir) == False:
            os.makedirs(self.ckptdir)
            
    def build(self, global_step=None):
        self._build()
        
        self.activations.append(self.x)
        self.lm.add_line(self.activations[-1])
        for layer in self.layers:
            activation = layer(self.activations[-1])
            self.activations.append(activation)
            self.lm.add_line(activation)
            
        self.y_hat = self.activations[-1]
        self.loss = self._loss()
        if global_step:
            self.opt_op = self.optimizer.minimize(self.loss, global_step=global_step)
        else:
            self.opt_op = self.optimizer.minimize(self.loss)
        
    def _build(self):
        return NotImplementedError
        
    def predict(self, x):
        y = self.sess.run([self.y_hat], feed_dict={self.x:x})[0]
        y = self.dm.denormalize(y)
        return y
        
    def _loss(self):
        return NotImplementedError
    
    def save(self):
        save_path = self.saver.save(self.sess, self.ckptdir)
        print ('{} SAVED IN {}'.format(self.name, save_path))
        
    def load(self):
        self.saver.restore(self.sess, self.ckptdir)
        print ('{} RESTORED FROM {}'.format(self.name, self.ckptdir))
        
    def terminate(self):
        self.sess.close()
        tf.compat.v1.reset_default_graph()
        print ('{} TERMINATED'.format(self.name))
