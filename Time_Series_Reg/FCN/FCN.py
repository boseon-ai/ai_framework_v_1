import sys
import tensorflow as tf
sys.path.append('/home/boseon/v_1/ai_framework_v_1/common/')

from tqdm import tqdm
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
    def __init__(self, log_dir, name, lr, data_manager, log_manager, din, dout, _lambda=0.0001, act_fn=tf.nn.relu):
        super().__init__(log_dir, name, lr, data_manager, log_manager)
        self.din = din
        self.dout = dout
        self._lambda = _lambda
        self.act_fn = act_fn

        self.x = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, 1, 12, 22], name='X')
        self.y = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, self.dout], name='Y')
        self.training = tf.compat.v1.placeholder_with_default(False, shape=[], name='BN_training')

        #update_ops = tf.compat.v1.get_collection(tf.GraphKeys.UPDATE_OPS)
        global_step = tf.Variable(0, trainable=False)
        num_batch = self.dm.data['tr_feeder'].num_batch
        self.decayed_lr = tf.train.exponential_decay(self.lr, global_step, decay_steps=num_batch, decay_rate=0.9, staircase=True)
        step_op = tf.assign_add(global_step, 1)
        with tf.control_dependencies([step_op]):
            self.optimizer = tf.compat.v1.train.AdamOptimizer(self.decayed_lr)

        self.lm.add_line('Optimizer: Adam')
        self.lm.add_line('Learning rate: {}, with decay rate of 0.9 for every epoch.'.format(self.lr))
        self.lm.add_line('Activation func: {}'.format(self.act_fn))
        
        self.build(global_step)
        # self.opt_op = tf.group([self.opt_op, update_ops]) <-- For batch normalization, but not helpful.
        self.saver = tf.compat.v1.train.Saver()
        self.sess.run(tf.compat.v1.global_variables_initializer())
        
    def _build(self):
        self.layers.append(CNNLayer(name     = 'cnn_layer_1', 
                                    fshape   = [1,4,22,512],
                                    pshape   = [1,1,4,1],
                                    act_fn   = self.act_fn))

        self.layers.append(CNNLayer(name     = 'cnn_layer_2', 
                                    fshape   = [1,3,512,512],
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
        #reg_loss = self.l2_reg()
        #loss = self.lr * (self.mse + reg_loss)
        loss = self.lr * self.mse
        return loss

    def l2_reg(self):
        _vars = tf.trainable_variables()
        reg = self._lambda * tf.add_n([ tf.nn.l2_loss(v) for v in _vars if 'bias' not in v.name ])
        return reg

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
        dict_rst = None
        final_mse = None

        cnt = 0
        max_v_loss = np.inf

        for i in range(0, epoch):
            train_iterator = self.dm.data['tr_feeder'].get_iterator()
            for j, (x, y) in tqdm(enumerate(train_iterator), desc="epoch_{}".format(i)):
                _, t_loss = self.sess.run([self.opt_op, self.loss], feed_dict={self.x:x, self.y:y, self.training:True})
            
            lr, v_loss = self.sess.run([self.decayed_lr, self.loss], feed_dict={self.x:self.dm.data['valid_x'], 
                                                                                self.y:self.dm.data['valid_y'],
                                                                                self.training:False})

            train_loss.append(t_loss)
            valid_loss.append(v_loss)

            dict_rst, mse = self.accuracy()
            self.lm.add_line('Epoch: {}, iter: {}, lr: {:.10f}, train loss: {:.10f}, valid loss: {:.10f}, test mse: {:.10f}'.format(i, j, lr, t_loss, v_loss, mse))

            if v_loss < max_v_loss:
                final_mse = mse
                max_v_loss = v_loss
                self.save()
            else:
                cnt += 1
                if cnt > patience:
                    self.lm.add_line('overfitted')
                    #break
            
        self.save()    
        return dict_rst, final_mse, train_loss, valid_loss
