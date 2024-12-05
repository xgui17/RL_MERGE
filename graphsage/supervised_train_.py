from __future__ import division
from __future__ import print_function

import os
import time
import tensorflow as tf
import numpy as np
import sklearn
from sklearn import metrics

from graphsage.supervised_models import SupervisedGraphsage
from graphsage.models import SAGEInfo
from graphsage.minibatch import NodeMinibatchIterator
from graphsage.neigh_samplers import UniformNeighborSampler, MLNeighborSampler, FastMLNeighborSampler
from graphsage.utils import load_data

from scipy import sparse 
import matplotlib.pyplot as plt
import pdb
from tensorflow.python.client import timeline

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS

tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
#core params..
flags.DEFINE_string('model', 'graphsage_mean', 'model names. See README for possible values.')  
flags.DEFINE_float('learning_rate', 0.01, 'initial learning rate.')
flags.DEFINE_string("model_size", "small", "Can be big or small; model specific def'ns")
flags.DEFINE_string('train_prefix', '', 'prefix identifying training data. must be specified.')
flags.DEFINE_string('model_prefix', '', 'model idx.')

# sampler param
flags.DEFINE_boolean('nonlinear_sampler', False, 'Where to use nonlinear sampler o.w. linear sampler')
flags.DEFINE_float('uniform_ratio', 0.6, 'In case of FastML sampling, the percentile of uniform sampling preceding the regressor sampling')

# left to default values in main experiments 
flags.DEFINE_integer('epochs', 10, 'number of epochs to train.')
flags.DEFINE_float('dropout', 0.0, 'dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0.0, 'weight for l2 loss on embedding matrix.')
flags.DEFINE_integer('max_degree', 128, 'maximum node degree.')
flags.DEFINE_integer('samples_1', 25, 'number of samples in layer 1')
flags.DEFINE_integer('samples_2', 10, 'number of samples in layer 2')
flags.DEFINE_integer('samples_3', 0, 'number of users samples in layer 3. (Only for mean model)')
flags.DEFINE_integer('dim_1', 128, 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_integer('dim_2', 128, 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_integer('dim_3', 128, 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_boolean('random_context', True, 'Whether to use random context or direct edges')
flags.DEFINE_integer('batch_size', 512, 'minibatch size.')
flags.DEFINE_boolean('sigmoid', False, 'whether to use sigmoid loss')
flags.DEFINE_integer('identity_dim', 0, 'Set to positive value to use identity embedding features of that dimension. Default 0.')

#logging, saving, validation settings etc.
flags.DEFINE_string('base_log_dir', '.', 'base directory for logging and saving embeddings')
flags.DEFINE_integer('validate_iter', 5000, "how often to run a validation minibatch.")
flags.DEFINE_integer('validate_batch_size', 512, "how many nodes per validation sample.")
flags.DEFINE_integer('gpu', 0, "which gpu to use.")
flags.DEFINE_integer('print_every', 5, "How often to print training info.")
flags.DEFINE_integer('max_total_steps', 10**10, "Maximum total number of iterations")
flags.DEFINE_boolean('timeline', False, 'export timeline')

os.environ["CUDA_VISIBLE_DEVICES"]=str(FLAGS.gpu)

GPU_MEM_FRACTION = 0.8

def calc_f1(y_true, y_pred):
    if not FLAGS.sigmoid:
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
    else:
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0
    #pdb.set_trace()
    return metrics.f1_score(y_true, y_pred, average="micro"), metrics.f1_score(y_true, y_pred, average="macro")

# Define model evaluation function
def evaluate(sess, model, minibatch_iter, size=None):
    t_test = time.time()
    feed_dict_val, labels = minibatch_iter.node_val_feed_dict(size)
    node_outs_val = sess.run([model.preds, model.loss], 
                        feed_dict=feed_dict_val)
    mic, mac = calc_f1(labels, node_outs_val[0])
    return node_outs_val[1], mic, mac, (time.time() - t_test)

def log_dir(sampler_model_name):
    log_dir = FLAGS.base_log_dir + "/output/sup-" + FLAGS.train_prefix.split("/")[-2] + '-' + FLAGS.model_prefix + '-' + sampler_model_name
    log_dir += "/{model:s}_{model_size:s}_{lr:0.4f}/".format(
            model=FLAGS.model,
            model_size=FLAGS.model_size,
            lr=FLAGS.learning_rate)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def sampler_log_dir():
    log_dir = FLAGS.base_log_dir + "/output/sampler-sup-" + FLAGS.train_prefix.split("/")[-2] + '-' + FLAGS.model_prefix
    log_dir += "/{model:s}_{model_size:s}_{lr:0.4f}/".format(
            model=FLAGS.model,
            model_size=FLAGS.model_size,
            lr=FLAGS.learning_rate)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def incremental_evaluate(sess, model, minibatch_iter, size, run_options=None, run_metadata=None, test=False):
    t_test = time.time()
    finished = False
    val_losses = []
    val_preds = []
    labels = []
    iter_num = 0
    finished = False

    while not finished:
        feed_dict_val, batch_labels, finished, _  = minibatch_iter.incremental_node_val_feed_dict(size, iter_num, test=test)
        
        if feed_dict_val.values()[0] != FLAGS.batch_size:
            break

        node_outs_val = sess.run([model.preds, model.loss], 
                         feed_dict=feed_dict_val, options=run_options, run_metadata=run_metadata)

        val_preds.append(node_outs_val[0])
        labels.append(batch_labels)
        val_losses.append(node_outs_val[1])
        iter_num += 1
    val_preds = np.vstack(val_preds)
    labels = np.vstack(labels)
    f1_scores = calc_f1(labels, val_preds)
    return np.mean(val_losses), f1_scores[0], f1_scores[1], (time.time() - t_test)

def construct_placeholders(num_classes):
    # Define placeholders
    placeholders = {
        'labels' : tf.placeholder(tf.float32, shape=(None, num_classes), name='labels'),
        'batch' : tf.placeholder(tf.int32, shape=(FLAGS.batch_size), name='batch1'),
        #'batch' : tf.placeholder(tf.int32, shape=(None), name='batch1'),
        'dropout': tf.placeholder_with_default(0., shape=(), name='dropout'),
        'batch_size' : tf.placeholder(tf.int32, name='batch_size'),
    }
    return placeholders

def train(train_data, test_data=None, sampler_name='Uniform'):

    G = train_data[0]
    features = train_data[1]
    id_map = train_data[2]
    class_map  = train_data[4]

    if isinstance(list(class_map.values())[0], list):
        num_classes = len(list(class_map.values())[0])
    else:
        num_classes = len(set(class_map.values()))

    if not features is None:
        # pad with dummy zero vector
        features = np.vstack([features, np.zeros((features.shape[1],))])

    context_pairs = train_data[3] if FLAGS.random_context else None
    placeholders = construct_placeholders(num_classes)
    minibatch = NodeMinibatchIterator(G, 
            id_map,
            placeholders, 
            class_map,
            num_classes,
            batch_size=FLAGS.batch_size,
            max_degree=FLAGS.max_degree, 
            context_pairs = context_pairs)
    adj_info_ph = tf.placeholder(tf.int32, shape=minibatch.adj.shape)
    adj_info = tf.Variable(adj_info_ph, trainable=False, name="adj_info")

    adj_shape = adj_info.get_shape().as_list()

#    loss_node = tf.SparseTensor(indices=np.empty((0,2), dtype=np.int64), values=[], dense_shape=[adj_shape[0], adj_shape[0]])
#    loss_node_count = tf.SparseTensor(indices=np.empty((0,2), dtype=np.int64), values=[], dense_shape=[adj_shape[0], adj_shape[0]])
# 
    # newly added for storing cost in each adj cell
#    loss_node = tf.Variable(tf.zeros([minibatch.adj.shape[0], minibatch.adj.shape[0]]), trainable=False, name="loss_node", dtype=tf.float32)
#    loss_node_count = tf.Variable(tf.zeros([minibatch.adj.shape[0], minibatch.adj.shape[0]]), trainable=False, name="loss_node_count", dtype=tf.float32)
    
    if FLAGS.model == 'mean_concat':
        # Create model
        
        if sampler_name == 'Uniform':
            sampler = UniformNeighborSampler(adj_info)
        elif sampler_name == 'ML':
            sampler = MLNeighborSampler(adj_info, features)
        elif sampler_name == 'FastML':
            sampler = FastMLNeighborSampler(adj_info, features)

        if FLAGS.samples_3 != 0:
            layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                                SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2),
                                SAGEInfo("node", sampler, FLAGS.samples_3, FLAGS.dim_3)]
        elif FLAGS.samples_2 != 0:
            layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                                SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]
        else:
            layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1)]
        '''        
        ### 3 layer test
        layer_infos = [SAGEInfo("node", sampler, 50, FLAGS.dim_2),
                                SAGEInfo("node", sampler, 25, FLAGS.dim_2),
                                SAGEInfo("node", sampler, 10, FLAGS.dim_2)]
 
        '''
        
        # modified
        model = SupervisedGraphsage(num_classes, placeholders, 
                                     features,
                                     adj_info,
                                     #loss_node,
                                     #loss_node_count,
                                     minibatch.deg,
                                     layer_infos,
                                     concat=True,
                                     model_size=FLAGS.model_size,
                                     sigmoid_loss = FLAGS.sigmoid,
                                     identity_dim = FLAGS.identity_dim,
                                     logging=True)
#
#        model = SupervisedGraphsage(num_classes, placeholders, 
#                                     features,
#                                     adj_info,
#                                     minibatch.deg,
#                                     layer_infos, 
#                                     model_size=FLAGS.model_size,
#                                     sigmoid_loss = FLAGS.sigmoid,
#                                     identity_dim = FLAGS.identity_dim,
#                                     logging=True)

    elif FLAGS.model == 'mean_add':
        # Create model
        
        if sampler_name == 'Uniform':
            sampler = UniformNeighborSampler(adj_info)
        elif sampler_name == 'ML':
            sampler = MLNeighborSampler(adj_info, features)
        elif sampler_name == 'FastML':
            sampler = FastMLNeighborSampler(adj_info, features)

        if FLAGS.samples_3 != 0:
            layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                                SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2),
                                SAGEInfo("node", sampler, FLAGS.samples_3, FLAGS.dim_3)]
        elif FLAGS.samples_2 != 0:
            layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                                SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]
        else:
            layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1)]
        '''        
        ### 3 layer test
        layer_infos = [SAGEInfo("node", sampler, 50, FLAGS.dim_2),
                                SAGEInfo("node", sampler, 25, FLAGS.dim_2),
                                SAGEInfo("node", sampler, 10, FLAGS.dim_2)]
 
        '''
        
        # modified
        model = SupervisedGraphsage(num_classes, placeholders, 
                                     features,
                                     adj_info,
                                     #loss_node,
                                     #loss_node_count,
                                     minibatch.deg,
                                     layer_infos,
                                     concat=False,
                                     model_size=FLAGS.model_size,
                                     sigmoid_loss = FLAGS.sigmoid,
                                     identity_dim = FLAGS.identity_dim,
                                     logging=True)
#
#        model = SupervisedGraphsage(num_classes, placeholders, 
#                                     features,
#                                     adj_info,
#                                     minibatch.deg,
#                                     layer_infos, 
#                                     model_size=FLAGS.model_size,
#                                     sigmoid_loss = FLAGS.sigmoid,
#                                     identity_dim = FLAGS.identity_dim,
#                                     logging=True)
 
    elif FLAGS.model == 'LRmean_add':
        # Create model
        
        if sampler_name == 'Uniform':
            sampler = UniformNeighborSampler(adj_info)
        elif sampler_name == 'ML':
            sampler = MLNeighborSampler(adj_info, features)
        elif sampler_name == 'FastML':
            sampler = FastMLNeighborSampler(adj_info, features)

        if FLAGS.samples_3 != 0:
            layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                                SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2),
                                SAGEInfo("node", sampler, FLAGS.samples_3, FLAGS.dim_3)]
        elif FLAGS.samples_2 != 0:
            layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                                SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]
        else:
            layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1)]
        '''        
        ### 3 layer test
        layer_infos = [SAGEInfo("node", sampler, 50, FLAGS.dim_2),
                                SAGEInfo("node", sampler, 25, FLAGS.dim_2),
                                SAGEInfo("node", sampler, 10, FLAGS.dim_2)]
 
        '''
        
        # modified
        model = SupervisedGraphsage(num_classes, placeholders, 
                                     features,
                                     adj_info,
                                     #loss_node,
                                     #loss_node_count,
                                     minibatch.deg,
                                     layer_infos,
                                     aggregator_type="LRmean",
                                     concat=False,
                                     model_size=FLAGS.model_size,
                                     sigmoid_loss = FLAGS.sigmoid,
                                     identity_dim = FLAGS.identity_dim,
                                     logging=True)
#
#        model = SupervisedGraphsage(num_classes, placeholders, 
#                                     features,
#                                     adj_info,
#                                     minibatch.deg,
#                                     layer_infos, 
#                                     model_size=FLAGS.model_size,
#                                     sigmoid_loss = FLAGS.sigmoid,
#                                     identity_dim = FLAGS.identity_dim,
#                                     logging=True)

    elif FLAGS.model == 'logicmean':
        # Create model
        
        if sampler_name == 'Uniform':
            sampler = UniformNeighborSampler(adj_info)
        elif sampler_name == 'ML':
            sampler = MLNeighborSampler(adj_info, features)
        elif sampler_name == 'FastML':
            sampler = FastMLNeighborSampler(adj_info, features)



        if FLAGS.samples_3 != 0:
            layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                                SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2),
                                SAGEInfo("node", sampler, FLAGS.samples_3, FLAGS.dim_2)]
        elif FLAGS.samples_2 != 0:
            layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                                SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]
        else:
            layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1)]
        '''        
        ### 3 layer test
        layer_infos = [SAGEInfo("node", sampler, 50, FLAGS.dim_2),
                                SAGEInfo("node", sampler, 25, FLAGS.dim_2),
                                SAGEInfo("node", sampler, 10, FLAGS.dim_2)]
 
        '''
        
        # modified
        model = SupervisedGraphsage(num_classes, placeholders, 
                                     features,
                                     adj_info,
                                     #loss_node,
                                     #loss_node_count,
                                     minibatch.deg,
                                     layer_infos,
                                     aggregator_type='logicmean',
                                     concat = True,
                                     model_size=FLAGS.model_size,
                                     sigmoid_loss = FLAGS.sigmoid,
                                     identity_dim = FLAGS.identity_dim,
                                     logging=True)
#
    elif FLAGS.model == 'attmean':
        # Create model
        
        if sampler_name == 'Uniform':
            sampler = UniformNeighborSampler(adj_info)
        elif sampler_name == 'ML':
            sampler = MLNeighborSampler(adj_info, features)
        elif sampler_name == 'FastML':
            sampler = FastMLNeighborSampler(adj_info, features)



        if FLAGS.samples_3 != 0:
            layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                                SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2),
                                SAGEInfo("node", sampler, FLAGS.samples_3, FLAGS.dim_2)]
        elif FLAGS.samples_2 != 0:
            layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                                SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]
        else:
            layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1)]
        '''        
        ### 3 layer test
        layer_infos = [SAGEInfo("node", sampler, 50, FLAGS.dim_2),
                                SAGEInfo("node", sampler, 25, FLAGS.dim_2),
                                SAGEInfo("node", sampler, 10, FLAGS.dim_2)]
 
        '''
        
        # modified
        model = SupervisedGraphsage(num_classes, placeholders, 
                                     features,
                                     adj_info,
                                     #loss_node,
                                     #loss_node_count,
                                     minibatch.deg,
                                     layer_infos,
                                     aggregator_type='attmean',
                                     model_size=FLAGS.model_size,
                                     sigmoid_loss = FLAGS.sigmoid,
                                     identity_dim = FLAGS.identity_dim,
                                     logging=True)
#
#        model = SupervisedGraphsage(num_classes, placeholders, 
#                                     features,
#                                     adj_info,
#                                     minibatch.deg,
#                                     layer_infos, 
#                                     model_size=FLAGS.model_size,
#                                     sigmoid_loss = FLAGS.sigmoid,
#                                     identity_dim = FLAGS.identity_dim,
#                                     logging=True)

    elif FLAGS.model == 'gcn':
        
        if sampler_name == 'Uniform':
            sampler = UniformNeighborSampler(adj_info)
        elif sampler_name == 'ML':
            sampler = MLNeighborSampler(adj_info, features)
        elif sampler_name == 'FastML':
            sampler = FastMLNeighborSampler(adj_info, features)

            
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, 2*FLAGS.dim_1),
                            SAGEInfo("node", sampler, FLAGS.samples_2, 2*FLAGS.dim_2)]
        
        model = SupervisedGraphsage(num_classes, placeholders, 
                                     features,
                                     adj_info,
                                     minibatch.deg,
                                     layer_infos=layer_infos, 
                                     aggregator_type="gcn",
                                     model_size=FLAGS.model_size,
                                     concat=False,
                                     sigmoid_loss = FLAGS.sigmoid,
                                     identity_dim = FLAGS.identity_dim,
                                     logging=True)

    elif FLAGS.model == 'graphsage_seq':
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                            SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]

        model = SupervisedGraphsage(num_classes, placeholders, 
                                     features,
                                     adj_info,
                                     minibatch.deg,
                                     layer_infos=layer_infos, 
                                     aggregator_type="seq",
                                     model_size=FLAGS.model_size,
                                     sigmoid_loss = FLAGS.sigmoid,
                                     identity_dim = FLAGS.identity_dim,
                                     logging=True)

    elif FLAGS.model == 'graphsage_maxpool':
        
        if sampler_name == 'Uniform':
            sampler = UniformNeighborSampler(adj_info)
        elif sampler_name == 'ML':
            sampler = MLNeighborSampler(adj_info, features)
        elif sampler_name == 'FastML':
            sampler = FastMLNeighborSampler(adj_info, features)


            
        #sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                            SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]

        model = SupervisedGraphsage(num_classes, placeholders, 
                                    features,
                                    adj_info,
                                    minibatch.deg,
                                     layer_infos=layer_infos, 
                                     aggregator_type="maxpool",
                                     model_size=FLAGS.model_size,
                                     sigmoid_loss = FLAGS.sigmoid,
                                     identity_dim = FLAGS.identity_dim,
                                     logging=True)

    elif FLAGS.model == 'graphsage_meanpool':
        
        if sampler_name == 'Uniform':
            sampler = UniformNeighborSampler(adj_info)
        elif sampler_name == 'ML':
            sampler = MLNeighborSampler(adj_info, features)
        elif sampler_name == 'FastML':
            sampler = FastMLNeighborSampler(adj_info, features)

    
            
        #sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                            SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]

        model = SupervisedGraphsage(num_classes, placeholders, 
                                    features,
                                    adj_info,
                                    minibatch.deg,
                                     layer_infos=layer_infos, 
                                     aggregator_type="meanpool",
                                     model_size=FLAGS.model_size,
                                     sigmoid_loss = FLAGS.sigmoid,
                                     identity_dim = FLAGS.identity_dim,
                                     logging=True)

    else:
        raise Exception('Error: model name unrecognized.')

    config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
    config.gpu_options.allow_growth = True
    #config.gpu_options.per_process_gpu_memory_fraction = GPU_MEM_FRACTION
    config.allow_soft_placement = True
    
    # Initialize session
    sess = tf.Session(config=config)
    merged = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(log_dir(sampler_name), sess.graph)
    
    # Save model
    saver = tf.train.Saver()
    model_path =  './model/' + FLAGS.train_prefix.split('/')[-1] + '-' + FLAGS.model_prefix + '-' + sampler_name
    model_path += "/{model:s}_{model_size:s}_{lr:0.4f}/".format(
            model=FLAGS.model,
            model_size=FLAGS.model_size,
            lr=FLAGS.learning_rate)

    if not os.path.exists(model_path):
        os.makedirs(model_path)


    # Init variables
    sess.run(tf.global_variables_initializer(), feed_dict={adj_info_ph: minibatch.adj})
   

    # Restore params of ML sampler model
    if sampler_name == 'ML' or sampler_name == 'FastML':
        sampler_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="MLsampler")
        #pdb.set_trace() 
        saver_sampler = tf.train.Saver(var_list=sampler_vars)
        sampler_model_path = './model/MLsampler-' + FLAGS.train_prefix.split('/')[-1] + '-' + FLAGS.model_prefix
        sampler_model_path += "/{model:s}_{model_size:s}_{lr:0.4f}/".format(
            model=FLAGS.model,
            model_size=FLAGS.model_size,
            lr=FLAGS.learning_rate)

        saver_sampler.restore(sess, sampler_model_path + 'model.ckpt')

    # Train model
    
    total_steps = 0
    avg_time = 0.0
    epoch_val_costs = []

    train_adj_info = tf.assign(adj_info, minibatch.adj)
    val_adj_info = tf.assign(adj_info, minibatch.test_adj)
    
    
    val_cost_ = []
    val_f1_mic_ = []
    val_f1_mac_ = []
    duration_ = []

    ln_acc = sparse.csr_matrix((adj_shape[0], adj_shape[0]), dtype=np.float32)
    lnc_acc = sparse.csr_matrix((adj_shape[0], adj_shape[0]), dtype=np.int32)
    
    ln_acc = ln_acc.tolil()
    lnc_acc = lnc_acc.tolil()
# 
#    ln_acc = np.zeros([adj_shape[0], adj_shape[0]])
#    lnc_acc = np.zeros([adj_shape[0], adj_shape[0]])


    for epoch in range(FLAGS.epochs): 
        minibatch.shuffle() 

        iter = 0
        print('Epoch: %04d' % (epoch + 1))
        epoch_val_costs.append(0)
        
        #for j in range(2):
        while not minibatch.end():
            # Construct feed dictionary
            feed_dict, labels = minibatch.next_minibatch_feed_dict()
            

            if feed_dict.values()[0] != FLAGS.batch_size:
                break
            

            feed_dict.update({placeholders['dropout']: FLAGS.dropout})

            t = time.time()
            # Training step
            #outs = sess.run([merged, model.opt_op, model.loss, model.preds], feed_dict=feed_dict)
           
            outs = sess.run([merged, model.opt_op, model.loss, model.preds, model.loss_node, model.loss_node_count, model.out_mean], feed_dict=feed_dict)
            train_cost = outs[2]


            if iter % FLAGS.validate_iter == 0:
                # Validation
                sess.run(val_adj_info.op)
                if FLAGS.validate_batch_size == -1:
                    val_cost, val_f1_mic, val_f1_mac, duration = incremental_evaluate(sess, model, minibatch, FLAGS.batch_size)
                else:
                    val_cost, val_f1_mic, val_f1_mac, duration = evaluate(sess, model, minibatch, FLAGS.validate_batch_size)
                
                # accumulate val results
                val_cost_.append(val_cost)
                val_f1_mic_.append(val_f1_mic)
                val_f1_mac_.append(val_f1_mac)
                duration_.append(duration)

                #
                sess.run(train_adj_info.op)
                epoch_val_costs[-1] += val_cost


            if total_steps % FLAGS.print_every == 0:
                summary_writer.add_summary(outs[0], total_steps)
    
            # Print results
            avg_time = (avg_time * total_steps + time.time() - t) / (total_steps + 1)

            # loss_node
            #import pdb
            #pdb.set_trace()
            
#            if epoch > 0.7*FLAGS.epochs:
#                ln = outs[-2].values
#                ln_idx = outs[-2].indices
#                ln_acc[ln_idx[:,0], ln_idx[:,1]] += ln
#           
#
#                lnc = outs[-1].values
#                lnc_idx = outs[-1].indices
#                lnc_acc[lnc_idx[:,0], lnc_idx[:,1]] += lnc
             
            
            ln = outs[4].values
            ln_idx = outs[4].indices
            ln_acc[ln_idx[:,0], ln_idx[:,1]] += ln
           

            lnc = outs[5].values
            lnc_idx = outs[5].indices
            lnc_acc[lnc_idx[:,0], lnc_idx[:,1]] += lnc
           
            #pdb.set_trace()
            #idx = np.where(lnc_acc != 0)
            #loss_node_mean = (ln_acc[idx[0], idx[1]]).mean()
            #loss_node_count_mean = (lnc_acc[idx[0], idx[1]]).mean()

            if total_steps % FLAGS.print_every == 0:
                train_f1_mic, train_f1_mac = calc_f1(labels, outs[3])
                print("Iter:", '%04d' % iter, 
                      "train_loss=", "{:.5f}".format(train_cost),
                      "train_f1_mic=", "{:.5f}".format(train_f1_mic), 
                      "train_f1_mac=", "{:.5f}".format(train_f1_mac), 
                      "val_loss=", "{:.5f}".format(val_cost),
                      "val_f1_mic=", "{:.5f}".format(val_f1_mic), 
                      "val_f1_mac=", "{:.5f}".format(val_f1_mac),
                      #"loss_node=", "{:.5f}".format(loss_node_mean),
                      #"loss_node_count=", "{:.5f}".format(loss_node_count_mean),
                      "time=", "{:.5f}".format(avg_time))
 

               
            iter += 1
            total_steps += 1

            if total_steps > FLAGS.max_total_steps:
                break

        if total_steps > FLAGS.max_total_steps:
                break
      

    # Save model
    save_path = saver.save(sess, model_path+'model.ckpt')
    print ('model is saved at %s'%save_path)


    # Save loss node and count
    loss_node_path = './loss_node/' + FLAGS.train_prefix.split('/')[-1] + '-' + FLAGS.model_prefix + '-' + sampler_name
    loss_node_path += "/{model:s}_{model_size:s}_{lr:0.4f}/".format(
            model=FLAGS.model,
            model_size=FLAGS.model_size,
            lr=FLAGS.learning_rate)
    if not os.path.exists(loss_node_path):
        os.makedirs(loss_node_path)

    loss_node = sparse.save_npz(loss_node_path + 'loss_node.npz', sparse.csr_matrix(ln_acc))
    loss_node_count = sparse.save_npz(loss_node_path + 'loss_node_count.npz', sparse.csr_matrix(lnc_acc))
    print ('loss and count per node is saved at %s'%loss_node_path)    
 
#    # save images of loss node and count
#    plt.imsave(loss_node_path + 'loss_node_mean.png', np.uint8(np.round(np.divide(ln_acc.todense()[:1024,:1024], lnc_acc.todense()[:1024,:1024]+1e-10))), cmap='jet', vmin=0, vmax=255)
#    plt.imsave(loss_node_path + 'loss_node_count.png', np.uint8(lnc_acc.todense()[:1024,:1024]), cmap='jet', vmin=0, vmax=255)
#

    print("Validation per epoch in training")
    for ep in range(FLAGS.epochs):
        print("Epoch: %04d"%ep, " val_cost={:.5f}".format(val_cost_[ep]), " val_f1_mic={:.5f}".format(val_f1_mic_[ep]), " val_f1_mac={:.5f}".format(val_f1_mac_[ep]), " duration={:.5f}".format(duration_[ep]))
    
    print("Optimization Finished!")
    sess.run(val_adj_info.op)

    # full validation 
    val_cost_ = []
    val_f1_mic_ = []
    val_f1_mac_ = []
    duration_ = []
    for iter in range(10):
        val_cost, val_f1_mic, val_f1_mac, duration = incremental_evaluate(sess, model, minibatch, FLAGS.batch_size)
        print("Full validation stats:",
                          "loss=", "{:.5f}".format(val_cost),
                          "f1_micro=", "{:.5f}".format(val_f1_mic),
                          "f1_macro=", "{:.5f}".format(val_f1_mac),
                          "time=", "{:.5f}".format(duration))

        val_cost_.append(val_cost)
        val_f1_mic_.append(val_f1_mic)
        val_f1_mac_.append(val_f1_mac)
        duration_.append(duration)
    
    print("mean: loss={:.5f} f1_micro={:.5f} f1_macro={:.5f} time={:.5f}\n".format(np.mean(val_cost_), np.mean(val_f1_mic_), np.mean(val_f1_mac_), np.mean(duration_)))
  
    # write validation results
    with open(log_dir(sampler_name) + "val_stats.txt", "w") as fp:
        for iter in range(10):
            fp.write("loss={:.5f} f1_micro={:.5f} f1_macro={:.5f} time={:.5f}\n".format(val_cost_[iter], val_f1_mic_[iter], val_f1_mac_[iter], duration_[iter]))
        
        fp.write("mean: loss={:.5f} f1_micro={:.5f} f1_macro={:.5f} time={:.5f}\n".format(np.mean(val_cost_), np.mean(val_f1_mic_), np.mean(val_f1_mac_), np.mean(duration_)))
        fp.write("variance: loss={:.5f} f1_micro={:.5f} f1_macro={:.5f} time={:.5f}\n".format(np.var(val_cost_), np.var(val_f1_mic_), np.var(val_f1_mac_), np.var(duration_)))
        

    # test 
    val_cost_ = []
    val_f1_mic_ = []
    val_f1_mac_ = []
    duration_ = []

    print("Writing test set stats to file (don't peak!)")
    
    # timeline
    if FLAGS.timeline == True:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
    else:
        run_options = None
        run_metadata = None

    for iter in range(10):
        
        val_cost, val_f1_mic, val_f1_mac, duration = incremental_evaluate(sess, model, minibatch, FLAGS.batch_size, run_options, run_metadata, test=True)
        
        #val_cost, val_f1_mic, val_f1_mac, duration = incremental_evaluate(sess, model, minibatch, FLAGS.batch_size, test=True)
        print("Full validation stats:",
                          "loss=", "{:.5f}".format(val_cost),
                          "f1_micro=", "{:.5f}".format(val_f1_mic),
                          "f1_macro=", "{:.5f}".format(val_f1_mac),
                          "time=", "{:.5f}".format(duration))

   
        val_cost_.append(val_cost)
        val_f1_mic_.append(val_f1_mic)
        val_f1_mac_.append(val_f1_mac)
        duration_.append(duration)



    print("mean: loss={:.5f} f1_micro={:.5f} f1_macro={:.5f} time={:.5f}\n".format(np.mean(val_cost_), np.mean(val_f1_mic_), np.mean(val_f1_mac_), np.mean(duration_)))
    
    # write test results
    with open(log_dir(sampler_name) + "test_stats.txt", "w") as fp:
        for iter in range(10):
            fp.write("loss={:.5f} f1_micro={:.5f} f1_macro={:.5f} time={:.5f}\n".
                        format(val_cost_[iter], val_f1_mic_[iter], val_f1_mac_[iter], duration_[iter]))
        
        fp.write("mean: loss={:.5f} f1_micro={:.5f} f1_macro={:.5f} time={:.5f}\n".
                        format(np.mean(val_cost_), np.mean(val_f1_mic_), np.mean(val_f1_mac_), np.mean(duration_)))
        fp.write("variance: loss={:.5f} f1_micro={:.5f} f1_macro={:.5f} time={:.5f}\n".
                        format(np.var(val_cost_), np.var(val_f1_mic_), np.var(val_f1_mac_), np.var(duration_)))

    
    # create timeline object, and write it to a json
    if FLAGS.timeline == True:
        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format(show_memory=True)
        with open(log_dir(sampler_name) + 'timeline.json', 'w') as f:
            print ('timeline written at %s'%(log_dir(sampler_name)+'timelnie.json'))
            f.write(ctf)

  
    sess.close()
    tf.reset_default_graph()

# Sampler
def train_sampler(train_data):

    features = train_data[1]
    #batch_size = FLAGS.batch_size
    batch_size = 512

    if not features is None:
        features = np.vstack([features, np.zeros((features.shape[1],))])
   
    # debug
    features = features[:,:50]

    
    node_size = len(features)
    node_dim = len(features[0])

    # build model
    # input (features of vertex and its neighbor, label)
    x1_ph = tf.placeholder(shape=[batch_size, node_dim], dtype=tf.float32)
    x2_ph = tf.placeholder(shape=[batch_size, node_dim], dtype=tf.float32) 
    y_ph = tf.placeholder(shape=[batch_size], dtype=tf.float32)
    
    with tf.variable_scope("MLsampler"):
        
        if FLAGS.nonlinear_sampler == True:
       
            print ("Non-linear regression sampler used")

            l = tf.layers.dense(tf.concat([x1_ph, x2_ph], axis=1), 1, activation=None, trainable=True, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='dense')
      
            out = tf.nn.relu(tf.exp(l), name='relu')
        else:

            print ("Linear regression sampler used")
           
            l = tf.layers.dense(x1_ph, node_dim, activation=None, trainable=True, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='dense')
            
            l = tf.matmul(l, x2_ph, transpose_b=True, name='matmul')
            out = tf.nn.relu(l, name='relu')
        ###

    loss = tf.nn.l2_loss(out-y_ph, name='loss')/batch_size
    
    '''
    with tf.variable_scope("MLsampler"):
        #bias = tf.Variable(tf.zeros([1]), trainable=True, name='bias')
        # layer 
        # relu(x1*W*x2)
       
        #drop_rate = 0.5
        #x1_ph = tf.layers.dropout(x1_ph, rate=drop_rate, training=True) 
        l = tf.layers.dense(x1_ph, node_dim, activation=None, trainable=True, name='dense')
        #l = tf.nn.relu(l, name='relu')

        #l = tf.layers.dense(l, node_dim, activation=None, trainable=True, name='dense2')
        #l = tf.nn.relu(l, name='relu')

        l = tf.matmul(l, x2_ph, transpose_b=True, name='matmul')
        
        #l = tf.nn.bias_add(l, tf.tile(bias, [batch_size]))
        out = tf.nn.relu(l, name='relu')
        ###

    loss = tf.nn.l2_loss(out-y_ph, name='loss')/batch_size
    '''
     
#    sampler_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="MLsampler")
#    for var in sampler_vars:
#        loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
#
    
    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate, name='Adam').minimize(loss)
    init = tf.global_variables_initializer()


    # configuration
    config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

  
    # load data
    loss_node_path = './loss_node/' + FLAGS.train_prefix.split('/')[-1] + '-' + FLAGS.model_prefix + '-Uniform'
    loss_node_path += "/{model:s}_{model_size:s}_{lr:0.4f}/".format(
            model=FLAGS.model,
            model_size=FLAGS.model_size,
            lr=FLAGS.learning_rate)

    loss_node = sparse.load_npz(loss_node_path + 'loss_node.npz')
    loss_node_count = sparse.load_npz(loss_node_path + 'loss_node_count.npz')

    #idx_nz = np.where(loss_node_count != 0)
    
    #pdb.set_trace()
    
    idx_nz = sparse.find(loss_node_count)
   
    # due to out of memory, select randomly limited number of data node
    vertex = features[idx_nz[0]]
    neighbor = features[idx_nz[1]]
    count = idx_nz[2]
    y = np.divide(sparse.find(loss_node)[2],count)


   # if FLAGS.train_prefix.split('/')[-1] == 'reddit':

   #     perm = np.random.permutation(idx_nz[0].shape[0])
   #     perm = perm[:200000]
   #     vertex = features[idx_nz[0][perm]]
   #     neighbor = features[idx_nz[1][perm]]
   #     count = idx_nz[2][perm]
   #     y = np.divide(sparse.find(loss_node)[2][perm],count)

   # else:

   #     vertex = features[idx_nz[0]]
   #     neighbor = features[idx_nz[1]]
   #     count = idx_nz[2]
   #     y = np.divide(sparse.find(loss_node)[2],count)

   
    # partition train/validation data
    vertex_tr = vertex[:-batch_size]
    neighbor_tr = neighbor[:-batch_size]
    y_tr = y[:-batch_size]

    vertex_val = vertex[-batch_size:] 
    neighbor_val = neighbor[-batch_size:]
    y_val = y[-batch_size:]

    iter_size = int(vertex_tr.shape[0]/batch_size)

    # initialize session
    sess = tf.Session(config=config)
    # summary
    tf.summary.scalar('loss', loss)
    merged_summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(sampler_log_dir(), sess.graph)


    # save model
    saver = tf.train.Saver()
    model_path = './model/MLsampler-' + FLAGS.train_prefix.split('/')[-1] + '-' + FLAGS.model_prefix
    model_path += "/{model:s}_{model_size:s}_{lr:0.4f}/".format(
            model=FLAGS.model,
            model_size=FLAGS.model_size,
            lr=FLAGS.learning_rate)

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    # init variables
    sess.run(init)
        
    # train
    total_steps = 0
    avg_time = 0.0
    
    #for epoch in range(50):
    for epoch in range(FLAGS.epochs):
        
        # shuffle
        perm = np.random.permutation(vertex_tr.shape[0])

        print("Epoch: %04d" %(epoch+1))

        for iter in range(iter_size):
                    
            # allocate batch
            vtr = vertex_tr[perm[iter*batch_size:(iter+1)*batch_size]]
            ntr = neighbor_tr[perm[iter*batch_size:(iter+1)*batch_size]]
            ytr = y_tr[perm[iter*batch_size:(iter+1)*batch_size]]

            t = time.time()
            outs = sess.run([ loss, optimizer, merged_summary_op], feed_dict={x1_ph: vtr, x2_ph: ntr, y_ph: ytr})
            train_loss = outs[0]
           

            # validation
            if iter%FLAGS.validate_iter == 0:

                outs = sess.run([ loss, optimizer, merged_summary_op], feed_dict={x1_ph: vertex_val, x2_ph: neighbor_val, y_ph: y_val})  
                val_loss = outs[0]

                
            avg_time = (avg_time*total_steps+time.time() - t)/(total_steps+1)

            # print 
            if total_steps%FLAGS.print_every == 0:
                print("Iter:", "%04d"%iter,
                                "train_loss=", "{:.5f}".format(train_loss),
                                "val_loss=", "{:.5f}".format(val_loss))

            total_steps +=1
            
            if total_steps > FLAGS.max_total_steps:
                break
        
        # save_model
        save_path = saver.save(sess, model_path+'model.ckpt')
        print ('model is saved at %s'%save_path)

    sess.close()
    tf.reset_default_graph()


def main(argv=None):

    ## train graphsage model
    print("Loading training data..")
    train_data = load_data(FLAGS.train_prefix)
    print("Done loading training data..")

    print("Start training uniform sampling + graphsage model..")
    #train(train_data, sampler_name='Uniform')
    print("Done training uniform sampling + graphsage model..")

    ## train sampler
    print("Start training ML sampler..")
    train_sampler(train_data)
    print("Done training ML sampler..")
    
    ## train 
    print("Start training ML sampling + graphsage model..")
    train(train_data, sampler_name='FastML')
    print("Done training ML sampling + graphsage model..")

if __name__ == '__main__':
    tf.app.run()
