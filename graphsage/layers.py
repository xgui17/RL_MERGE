from __future__ import division
from __future__ import print_function


# DISCLAIMER:
# Boilerplate parts of this code file were originally forked from
# https://github.com/tkipf/gcn
# which itself was very inspired by the keras package

# global unique layer ID dictionary for layer name assignment
import tensorflow as tf


# Mimic FLAGS object
class FLAGS:
    weight_decay = 0.0

def set_layer_flags(args):
    FLAGS.weight_decay = args.weight_decay


_LAYER_UIDS = {}

def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]

class Layer:
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging', 'model_size'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        self.logging = kwargs.get('logging', False)
        self.sparse_inputs = False

    def _call(self, *args, **kwargs):
        return args, kwargs

    def __call__(self, *args, **kwargs):
        with tf.name_scope(self.name):
            outputs = self._call(*args, **kwargs)
            return outputs

class Dense(Layer):
    def __init__(self, input_dim, output_dim, dropout=0.,
                 act=tf.nn.relu, bias=True, **kwargs):
        super(Dense, self).__init__(**kwargs)
        self.dropout = dropout
        self.act = act
        self.bias = bias
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.vars['weights'] = tf.Variable(
            initial_value=tf.keras.initializers.GlorotUniform()(shape=(input_dim, output_dim)),
            dtype=tf.float32,
            name=self.name + '_weights'
        )
        if self.bias:
            self.vars['bias'] = tf.Variable(
                initial_value=tf.zeros([output_dim]),
                dtype=tf.float32,
                name=self.name + '_bias'
            )

    def _call(self, inputs):
        x = tf.nn.dropout(inputs, rate=self.dropout)
        output = tf.matmul(x, self.vars['weights'])
        if self.bias:
            output += self.vars['bias']
        return self.act(output)

