import tensorflow as tf

def init_weight(name, shape, init='he'):
    if init == 'he':
        initializer = tf.contrib.layers.variance_scaling_initializer()
    elif init == 'xavier':
        initializer = tf.contrib.layers.xavier_initializer()
    else:
        raise ValueError('choose initializer ==> he or xavier')
        
    return tf.compat.v1.get_variable(name=name, shape=shape, dtype=tf.float32, initializer=initializer)

def init_bias(name, shape):
    return tf.compat.v1.get_variable(name = name, shape=shape, initializer=tf.constant_initializer(0.0))
    
class DenseLayer:
    def __init__(self, name, din, dout, act_fn):
        self.name = name
        self.din = din
        self.dout = dout
        self.act_fn = act_fn
        self.vars = {}
        
        with tf.compat.v1.variable_scope(self.name):
            self.vars['weights'] = init_weight(name='weights', shape=[self.din, self.dout])
            self.vars['bias'] = init_bias(name='bias', shape=[self.dout])
        
    def __call__(self, x):
        out = tf.matmul(x, self.vars['weights']) + self.vars['bias']
        return self.act_fn(out)
    
class CNNLayer:
    def __init__(self, name, fshape, pshape, act_fn, strides=[1,1,1,1], padding='SAME', flatten=False):
        self.name    = name
        self.act_fn  = act_fn
        self.strides = strides
        self.padding = padding
        self.fshape  = fshape
        self.pshape  = pshape
        self.flatten = flatten
        self.vars = {}
        
        with tf.compat.v1.variable_scope(self.name + '_vars'):
            self.vars['weights'] = init_weight(name='weights', shape=self.fshape)
            self.vars['bias'] = init_bias(name='bias', shape=self.fshape[3])
            
    def __call__(self, x):
        out = tf.nn.conv2d(x, self.vars['weights'], strides=self.strides, padding=self.padding)
        out = self.act_fn(out + self.vars['bias'])
        out = tf.nn.max_pool(out, ksize=self.pshape, strides=self.pshape, padding=self.padding)
        if self.flatten == True:
            out = tf.reshape(out, shape=[-1, self.fshape[3]])
            
        return out
