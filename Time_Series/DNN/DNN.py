import sys
import tensorflow as tf
sys.path.append('/home/boseon/projects/kewp_3/common/')

from layer import *
from model import *

class CustomizedDenseLayer(DenseLayer):
    def __init__(self, name, din, dout, act_fn, kr):
        super().__init__(name, din, dout, act_fn)
        self.keep_rate = kr
        
    def __call__(self, x):
        out = tf.matmul(x, self.vars['weights']) + self.vars['bias']
        out = self.act_fn(out)
        return tf.nn.dropout(out, self.keep_rate)

class DNN(Model):
    def __init__(self, log_dir, name, lr, data_manager, log_manager, din, dout, act_fn=tf.nn.relu):
        super().__init__(log_dir, name, lr, data_manager, log_manager)
        self.din = din
        self.dout = dout
        self.act_fn = act_fn

        self.x = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, self.din], name='X')
        self.y = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, self.dout], name='Y')
        self.kr = tf.compat.v1.placeholder(dtype=tf.float32)
        self.optimizer = tf.compat.v1.train.AdamOptimizer(self.lr)

        self.lm.add_line('Optimizer: Adam')
        self.lm.add_line('Learning rate: {}, w/o decay'.format(self.lr))
        self.lm.add_line('Activation func: {}'.format(self.act_fn))

        self.build()
        self.saver = tf.compat.v1.train.Saver()
        self.sess.run(tf.compat.v1.global_variables_initializer())

    def _build(self):
        self.layers.append(CustomizedDenseLayer(name='dense_1', din=self.din,  dout=500, act_fn=self.act_fn, kr=self.kr))
        self.layers.append(CustomizedDenseLayer(name='dense_2', din=500, dout=500, act_fn=self.act_fn, kr=self.kr))
        self.layers.append(CustomizedDenseLayer(name='dense_3', din=500, dout=500, act_fn=self.act_fn, kr=self.kr))
        self.layers.append(CustomizedDenseLayer(name='output',  din=500, dout=self.dout, act_fn=lambda x: x, kr=self.kr))

    def _loss(self):
        self.mse = tf.compat.v1.losses.mean_squared_error(self.y, self.y_hat)
        loss = self.lr * self.mse
        return loss
    
    def predict(self, x):
        y = self.sess.run([self.y_hat], feed_dict={self.x:x, self.kr:1.0})[0]
        y = self.dm.denormalize(y)
        return y

    def accuracy(self):
        dict_rst = {}
        y_hat, mse = self.sess.run([self.y_hat, self.mse], feed_dict={self.x:self.dm.data['test_x'], self.y:self.dm.data['test_y'], self.kr:1.0})
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
                _, t_loss = self.sess.run([self.opt_op, self.loss], feed_dict={self.x:x, self.y:y, self.kr:0.5})

                if j % 100 == 0:
                    v_loss = self.sess.run(self.loss, feed_dict={self.x:self.dm.data['valid_x'], self.y:self.dm.data['valid_y'], self.kr:1.0})
                    train_loss.append(t_loss)
                    valid_loss.append(v_loss)
                    print ('Epoch: {}, iteration: {}, train loss: {}, valid loss: {}'.format(i, j, t_loss, v_loss))


                    if v_loss < max_v_loss:
                        max_v_loss = v_loss
                        self.save()

            dict_rst, mse = self.accuracy()
            self.lm.add_line('Epoch: {}, MSE: {}'.format(i, mse))

        return train_loss, valid_loss














