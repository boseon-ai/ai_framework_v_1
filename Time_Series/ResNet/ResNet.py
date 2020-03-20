import sys
import tensorflow as tf
sys.path.append('/home/jovyan/common/')

from layer import *
from model import * 

class ResNetLayer:
    def __init__(self, name, 
                 fh=1, fw=[8,5,3], cin=22, cout=64, 
                 strides=[1,1,1,1], padding='SAME', 
                 act_fn=tf.nn.relu, training=None):
        self.name = name
        self.fh = fh
        self.fw = fw
        self.cin = cin
        self.cout = cout
        self.strides = strides
        self.padding = padding
        self.act_fn = act_fn
        self.training = training
        self.vars = {}

        with tf.compat.v1.variable_scope(self.name+'_vars'):
            for i in range(len(self.fw)):
                key = 'weight_'+str(i+1)
                self.vars[key] = init_weight(name=key, shape=[1,self.fw[i],cin,self.cout])
                cin = self.cout

            if self.cin != self.cout:
                key = 'weight_0'
                self.vars[key] = init_weight(name=key, shape=[1,1,self.cin,self.cout])

    def __call__(self, x):
        momentum=0.9
        out_0 = self.batch_normalization(x, self.training, momentum=momentum)

        out_1 = self.conv(out_0, self.vars['weight_1'])
        #out_1 = self.batch_normalization(out_1, self.training, momentum=momentum)
        out_1 = self.activation(out_1)

        out_2 = self.conv(out_1, self.vars['weight_2'])
        #out_2 = self.batch_normalization(out_2, self.training, momentum=momentum)
        out_2 = self.activation(out_2)

        out_3 = self.conv(out_2, self.vars['weight_3'])
        #out_3 = self.batch_normalization(out_3, self.training, momentum=momentum)

        if self.cin != self.cout:
            _x = self.conv(x, self.vars['weight_0'])
        else: 
            _x = x

        out = tf.add(_x, out_3)
        out = self.activation(out)
        return out

    def conv(self, x, filters):
        return tf.nn.conv2d(x, filters, self.strides, self.padding)

    def activation(self, x):
        return self.act_fn(x)

    def batch_normalization(self, x, training, momentum=0.9):
        return tf.compat.v1.layers.batch_normalization(x, training=training, momentum=momentum)

class ResNet(Model):
    def __init__(self, log_dir, name, lr, data_manager, log_manager, din, dout, filters, act_fn=tf.nn.relu):
        super().__init__(log_dir, name, lr, data_manager, log_manager)
        self.din = din
        self.dout = dout
        self.filters = filters
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
        self.layers.append(ResNetLayer(name='Layer_1', 
                                       fh=1, 
                                       fw=[8,5,3], 
                                       cin=22, 
                                       cout=self.filters[0], 
                                       strides=[1,1,1,1], 
                                       padding='SAME', 
                                       act_fn=self.act_fn, 
                                       training=self.training))

        self.layers.append(ResNetLayer(name='Layer_2', 
                                       fh=1, 
                                       fw=[8,5,3], 
                                       cin=self.filters[0], 
                                       cout=self.filters[1], 
                                       strides=[1,1,1,1], 
                                       padding='SAME', 
                                       act_fn=self.act_fn, 
                                       training=self.training))

        self.layers.append(ResNetLayer(name='Layer_3', 
                                       fh=1, 
                                       fw=[8,5,3], 
                                       cin=self.filters[1], 
                                       cout=self.filters[2], 
                                       strides=[1,1,1,1], 
                                       padding='SAME', 
                                       act_fn=self.act_fn, 
                                       training=self.training))

        self.layers.append(CNNLayer(name='output',
                                    fshape=[1,12,128,self.dout], 
                                    pshape=[1,1,1,1],
                                    act_fn=lambda x: x,
                                    padding='VALID',
                                    flatten=True))


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
    
    def train(self, epoch):
        train_loss = []
        valid_loss = []
        
        max_v_loss = np.inf

        for i in range(0, epoch):
            train_iterator = self.dm.data['tr_feeder'].get_iterator()
            for j, (x, y) in enumerate(train_iterator):
                _, t_loss = self.sess.run([self.opt_op, self.loss], feed_dict={self.x:x, self.y:y, self.training:True})
                
                if j % 100 == 0:
                    v_loss = self.sess.run(self.loss, feed_dict={self.x:self.dm.data['valid_x'], 
                                                                 self.y:self.dm.data['valid_y'],
                                                                 self.training:False})
                    
                    train_loss.append(t_loss)
                    valid_loss.append(v_loss)
                    
                    print ('Epoch: {}, iteration: {}, train loss: {}, valid loss: {}'.format(i, j, t_loss, v_loss))
                    
                    if v_loss < max_v_loss:
                        max_v_loss = v_loss
                        self.save()

            dict_rst, mse = self.accuracy()
            self.lm.add_line('Epoch: {}, MSE: {}'.format(i, mse))
            
        return train_loss, valid_loss
