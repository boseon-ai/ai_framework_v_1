import sys
import tensorflow as tf
sys.path.append('/home/jovyan/common/')

from layer_v_2 import *
from model import * 

class CustomizedCNNLayer(CNNLayer):
    def __init__(self, name, fshape, pshape, act_fn, strides=[1,1,1,1], padding='SAME', flatten=False, training=False, momentum=0.9):
        super().__init__(name, fshape, pshape, act_fn, strides, padding, flatten)
        self.training = training
        self.momentum = momentum
        
    def __call__(self, x):
        out = self.conv2d(x)
        out = self.batch_normalization(out, self.training, self.momentum)
        out = self.act_fn(out)
        out = self.pooling(out)
        if self.flatten == True:
            out = tf.reshape(out, shape=[-1, self.fshape[3]])
      
        return out
        
    def batch_normalization(self, x, training, momentum=0.9):
        return tf.compat.v1.layers.batch_normalization(x, training=training, momentum=momentum)

class FCN(Model):
    def __init__(self, log_dir, name, lr, data_manager, log_manager, din, dout, act_fn=tf.nn.relu):
        super().__init__(log_dir, name, lr, data_manager, log_manager)
        self.din = din
        self.dout = dout
        self.act_fn = act_fn

        self.x = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, 1, 12, 22], name='X')
        self.y = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, self.dout], name='Y')
        self.training = tf.compat.v1.placeholder_with_default(False, shape=[], name='BN_training')

        update_ops = tf.compat.v1.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.optimizer = tf.compat.v1.train.AdamOptimizer(self.lr)

        self.lm.add_line('Optimizer: Adam')
        self.lm.add_line('Learning rate: {}, w/o decay'.format(self.lr))
        self.lm.add_line('Activation func: {}'.format(self.act_fn))
        
        self.build()
        self.opt_op = tf.group([self.opt_op, update_ops])
        self.saver = tf.compat.v1.train.Saver()
        self.sess.run(tf.compat.v1.global_variables_initializer())
        
    def _build(self):
        self.layers.append(CNNLayer(name     = 'cnn_layer_1', 
                                    fshape   = [1,4,22,256],
                                    pshape   = [1,1,4,1],
                                    act_fn   = self.act_fn))

        self.layers.append(CNNLayer(name     = 'cnn_layer_2', 
                                    fshape   = [1,3,256,512],
                                    pshape   = [1,1,3,1],
                                    act_fn   = self.act_fn))
        
        self.layers.append(CNNLayer(name     = 'cnn_layer_3', 
                                    fshape   = [1,3,512,512],
                                    pshape   = [1,1,3,1],
                                    act_fn   = self.act_fn))

        self.layers.append(CNNLayer(name     = 'output_layer',
                                    fshape   = [1,1,512,self.dout],
                                    pshape   = [1,1,1,1],
                                    act_fn   = lambda x: x,
                                    flatten  = True))

    def _loss(self):
        self.mse = tf.compat.v1.losses.mean_squared_error(self.y, self.y_hat)
        loss = self.lr * self.mse
        return loss

    def accuracy(self):
        dict_rst = {}
        y_hat, mse = self.sess.run([self.y_hat, self.mse], 
                                   feed_dict={self.x:self.dm.data['test_x'], 
                                              self.y:self.dm.data['test_y'],
                                              self.training:False})
        
        y_hat = self.dm.denormalize(y_hat)
        y = self.dm.denormalize(self.dm.data['test_y'])
        dict_rst[self.name] = y_hat
        dict_rst['labels'] = y
        return dict_rst, mse
    
    def train(self, epoch, patience=5):
        train_loss = []
        valid_loss = []
        
        cnt = 0
        max_v_loss = np.inf

        for i in range(0, epoch):
            train_iterator = self.dm.data['tr_feeder'].get_iterator()
            for j, (x, y) in enumerate(train_iterator):
                _, t_loss = self.sess.run([self.opt_op, self.loss], feed_dict={self.x:x, self.y:y, self.training:True})
                
            
            v_loss = self.sess.run(self.loss, feed_dict={self.x:self.dm.data['valid_x'], 
                                                         self.y:self.dm.data['valid_y'],
                                                         self.training:False})

            train_loss.append(t_loss)
            valid_loss.append(v_loss)

            print ('Epoch: {}, train loss: {}, valid loss: {}'.format(i, t_loss, v_loss))

            if v_loss < max_v_loss:
                max_v_loss = v_loss
                self.save()
            else:
                cnt += 1
                if cnt > patience:
                    print ('overfitted')
                    break

            dict_rst, mse = self.accuracy()
            self.lm.add_line('Epoch: {}, test loss (mse): {}'.format(i, mse))
            
        self.save()    
        return train_loss, valid_loss
