from __future__ import division
from __future__ import print_function

from graphsage.layers import Layer

import tensorflow as tf
import torch



class FLAGS:
    learning_rate = 0.001
    weight_decay = 5e-4

def set_sampler_flags(args):
    FLAGS.learning_rate = args.learning_rate
    FLAGS.weight_decay = args.weight_decay
    FLAGS.max_degree = args.max_degree

import numpy as np
import pdb
"""
Classes that are used to sample node neighborhoods
"""

class UniformNeighborSampler(Layer):
    """
    Uniformly samples neighbors.
    Assumes that adj lists are padded with random re-sampling
    """
    def __init__(self, adj_info, **kwargs):
        super(UniformNeighborSampler, self).__init__(**kwargs)
        self.adj_info = adj_info

    def _call(self, inputs):
        ids, num_samples = inputs
        
        # get matrix of [numofids, degree(128)]
        adj_lists = tf.nn.embedding_lookup(self.adj_info, ids) 
        # shuffling along degree axis 
        #adj_lists = tf.transpose(tf.random_shuffle(tf.transpose(adj_lists)))
        # pick [numofids, num_samples]
        #adj_lists = tf.slice(adj_lists, [0,0], [-1, num_samples])
       
        unif_rand = tf.random.uniform([num_samples], minval=0, maxval=np.int(adj_lists.shape[1]), dtype=tf.int32)
        
        adj_lists = tf.gather(adj_lists, unif_rand, axis=1)
        

        condition = tf.equal(adj_lists, self.adj_info.shape[0]-1)
        case_true = tf.zeros(adj_lists.shape, tf.float32)
        case_false = tf.ones(adj_lists.shape, tf.float32)
        adj_lists_numnz = tf.reduce_sum(tf.where(condition, case_true, case_false), axis=1)
 
        att_lists = tf.ones(adj_lists.shape)
        dummy_ = adj_lists_numnz

        return adj_lists, att_lists, adj_lists_numnz, dummy_
        #return adj_lists

def sparse_gather(sparse_tensor, indices):
    """
    Gather rows from a SparseTensor corresponding to the given indices.
    """
    gathered_indices = tf.gather(sparse_tensor.indices, indices)
    gathered_values = tf.gather(sparse_tensor.values, indices)
    dense_shape = sparse_tensor.dense_shape
    return tf.sparse.SparseTensor(gathered_indices, gathered_values, dense_shape)


class MLNeighborSampler:
    """
    Sampling by regressor trained by RL-learning.
    Handles sparse adjacency matrices efficiently.
    """

    def __init__(self, adj_info, features, max_degree=64, nonlinear_sampler=True):
        """
        Initialize the sampler with adjacency matrix and features.
        :param adj_info: Sparse adjacency matrix as a SparseTensor
        :param features: Node feature matrix (dense)
        :param max_degree: Maximum number of neighbors to sample
        :param nonlinear_sampler: Whether to use a nonlinear sampling mechanism
        """
        self.adj_info = adj_info  # Should be a SparseTensor
        self.features = tf.constant(features, dtype=tf.float32, name="features")
        self.max_degree = max_degree
        self.node_dim = features.shape[1]
        self.nonlinear_sampler = nonlinear_sampler

    def _sparse_gather(self, sparse_tensor, indices):
        """
        Gather rows from a SparseTensor corresponding to the given indices.
        :param sparse_tensor: A SparseTensor
        :param indices: A tensor of row indices to gather
        :return: Filtered SparseTensor
        """
        indices = tf.cast(indices, sparse_tensor.indices.dtype)  # Ensure dtype compatibility

        # Create a mask for each row index in sparse_tensor.indices[:, 0]
        gathered_indices = []
        gathered_values = []
        for index in indices:
            mask = tf.equal(sparse_tensor.indices[:, 0], index)
            filtered_indices = tf.boolean_mask(sparse_tensor.indices, mask)
            filtered_values = tf.boolean_mask(sparse_tensor.values, mask)
            gathered_indices.append(filtered_indices)
            gathered_values.append(filtered_values)

        # Concatenate results
        final_indices = tf.concat(gathered_indices, axis=0)
        final_values = tf.concat(gathered_values, axis=0)

        # The dense shape remains unchanged
        dense_shape = sparse_tensor.dense_shape

        # Return the filtered SparseTensor
        return tf.sparse.SparseTensor(final_indices, final_values, dense_shape)
    

    def convert_tf_sparse_to_edge_index(self, tf_sparse):
        """
        Convert TensorFlow SparseTensor to PyTorch-compatible edge_index format.
        :param tf_sparse: TensorFlow SparseTensor
        :return: edge_index (PyTorch tensor) and edge_weight (PyTorch tensor)
        """
        # Convert indices and values to numpy
        indices = tf_sparse.indices.numpy()  # Shape: [num_nonzero, 2]
        values = tf_sparse.values.numpy()    # Shape: [num_nonzero]

        # Extract source and target from indices
        edge_index = torch.tensor(indices.T, dtype=torch.long)  # Shape: [2, num_edges]
        edge_weight = torch.tensor(values, dtype=torch.float32)  # Optional: weights

        return edge_index, edge_weight






    def __call__(self, ids, num_samples):
        # Gather adjacency lists for input node IDs
        adj_lists = self._sparse_gather(self.adj_info, ids)

        edge_index, edge_weight = self.convert_tf_sparse_to_edge_index(adj_lists)

        # Extract neighbor IDs and features
        neighbor_ids = adj_lists.indices[:, 1]
        neighbor_features = tf.nn.embedding_lookup(self.features, neighbor_ids)

        # Calculate the number of neighbors per node
        num_neighbors_per_node = tf.shape(neighbor_ids)[0] // tf.shape(ids)[0]
        print(f"Number of neighbors per node: {num_neighbors_per_node}")

        # Pad neighbor features if necessary
        if num_neighbors_per_node < self.max_degree:
            padding_needed = self.max_degree * tf.shape(ids)[0] - tf.shape(neighbor_features)[0]
            neighbor_features = tf.pad(
                neighbor_features,
                paddings=[[0, padding_needed], [0, 0]],
                mode='CONSTANT',
                constant_values=0.0
            )
            print(f"Neighbor features shape after padding: {tf.shape(neighbor_features)}")

        # Reshape neighbors
        gathered_neighbors = tf.reshape(
            neighbor_features, shape=[-1, self.max_degree, self.node_dim]
        )
        print(f"Gathered neighbors shape after reshaping: {tf.shape(gathered_neighbors)}")

        # Compute attention scores or sampling probabilities
        node_features = tf.nn.embedding_lookup(self.features, ids)
        tiled_node_features = tf.tile(tf.expand_dims(node_features, 1), [1, self.max_degree, 1])
        combined_features = tf.concat([tiled_node_features, gathered_neighbors], axis=-1)
        scores = tf.keras.layers.Dense(units=1, activation=tf.nn.relu)(combined_features)
        sampling_weights = tf.nn.softmax(scores, axis=1)

        # Squeeze sampling_weights to [batch_size, max_degree]
        sampling_weights = tf.squeeze(sampling_weights, axis=-1)

        # Handle cases where the number of neighbors is less than num_samples
        actual_neighbors = tf.shape(sampling_weights)[1]
        print(f"Actual neighbors in sampling_weights: {actual_neighbors}")
        print(f"Requested num_samples: {num_samples}")
        num_samples = tf.minimum(num_samples, actual_neighbors)

        # Select top-k neighbors
        top_k_scores, top_k_indices = tf.nn.top_k(sampling_weights, k=num_samples)
        selected_neighbors = tf.gather(gathered_neighbors, top_k_indices, batch_dims=1)

        # Debugging: Final outputs
        print(f"Selected neighbors shape: {tf.shape(selected_neighbors)}")
        print(f"Top-k scores shape: {tf.shape(top_k_scores)}")
        print(f"Top-k indices shape: {tf.shape(top_k_indices)}")

        # Compute other required outputs
        adj_lists_numnz = tf.reduce_sum(
            tf.cast(tf.not_equal(selected_neighbors, 0), dtype=tf.float32),
            axis=1
        )
        att_lists = top_k_scores

        return selected_neighbors, att_lists, adj_lists_numnz, edge_index, edge_weight





    






class FastMLNeighborSampler(Layer):
    
    """
    Fast ver. of Sampling by regressor trained by RL-learning
    Replaced sorting operation with batched arg operation
    
    """

    def __init__(self, adj_info, features, **kwargs):
        super(FastMLNeighborSampler, self).__init__(**kwargs)
        self.adj_info = adj_info
        self.features = tf.Variable(tf.constant(features, dtype=tf.float32), trainable=False)
        self.batch_size = FLAGS.max_degree
        self.node_dim = features.shape[1]
        self.reuse = False 

    def _call(self, node_ids, num_samples=None):
       
        ids = node_ids

        adj_lists = tf.nn.embedding_lookup(self.adj_info, ids)
        neig_num = np.int(self.adj_info.shape[1])
        
        #unif_rand = tf.random.uniform([np.int(neig_num/num_samples)*num_samples], minval=0, maxval=np.int(neig_num), dtype=tf.int32)
        #adj_lists = tf.gather(adj_lists, unif_rand, axis=1)
        
        
        adj_lists = tf.slice(adj_lists, [0,0], [-1, np.int(neig_num/num_samples)*num_samples])

        vert_num = np.int(adj_lists.shape[0])
        neig_num = np.int(adj_lists.shape[1])
 
        # build model 
        # l = W*x1
        # l = relu(l*x2^t)
        with tf.variable_scope("MLsampler"):

           
            v_f = tf.nn.embedding_lookup(self.features, ids)
            n_f = tf.nn.embedding_lookup(self.features, tf.reshape(adj_lists, [-1]))
            
            # debug
            node_dim = np.int(v_f.shape[1])
          
            n_f = tf.reshape(n_f, shape=[-1, neig_num, node_dim])

            
            if FLAGS.nonlinear_sampler == True:
            
                v_f = tf.tile(tf.expand_dims(v_f, axis=1), [1, neig_num, 1])
           
                l = tf.layers.dense(tf.concat([v_f, n_f], axis=2), 1, activation=tf.nn.relu, trainable=False, reuse=self.reuse, name='dense')
           
                #out = tf.nn.relu(tf.exp(l), name='relu')
                out = tf.exp(l)
                
            else:

                l = tf.layers.dense(v_f, node_dim, activation=None, trainable=False, reuse=self.reuse, name='dense')
           
                l = tf.expand_dims(l, axis=1)
                l = tf.matmul(l, n_f, transpose_b=True, name='matmul') 
           
                out = tf.nn.relu(l, name='relu')

            
            out = tf.squeeze(out)

            
            # group min
            group_dim = np.int(neig_num/num_samples)
            out = tf.reshape(out, [vert_num, num_samples, group_dim])
            idx_y = tf.argmin(out, axis=-1, output_type=tf.int32)
            #idx_y = tf.squeeze(tf.nn.top_k(-out, k=1)[1])
            delta = tf.expand_dims(tf.range(0, group_dim*num_samples, group_dim), axis=0)
            delta = tf.tile(delta, [vert_num, 1])
            idx_y = idx_y + delta


            idx_x = tf.range(vert_num)
            idx_x = tf.tile(tf.expand_dims(idx_x, -1), [1, num_samples])

            adj_lists = tf.gather_nd(adj_lists, tf.stack([tf.reshape(idx_x,[-1]), tf.reshape(idx_y,[-1])], axis=1))
            adj_lists = tf.reshape(adj_lists, [vert_num, num_samples])
            
            condition = tf.equal(adj_lists, self.adj_info.shape[0]-1)
            case_true = tf.zeros(adj_lists.shape, tf.float32)
            case_false = tf.ones(adj_lists.shape, tf.float32)
            adj_lists_numnz = tf.reduce_sum(tf.where(condition, case_true, case_false), axis=1)
           
            #out = tf.exp(out)
            #norm = tf.tile(tf.expand_dims(tf.reduce_sum(out, axis=1), -1), [1, num_samples])
            #att = tf.div(out, norm)
            
            #att = tf.nn.softmax(out,axis=1)
            #att_lists = tf.reshape(1+att, [vert_num, num_samples])  
            att_lists = tf.ones(adj_lists.shape)
            dummy_ = adj_lists_numnz            
            
            self.reuse = True

        #return adj_lists
        return adj_lists, att_lists, adj_lists_numnz, dummy_