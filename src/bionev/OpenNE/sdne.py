# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

__author__ = "Wang Binlu"
__email__ = "wblmail@whu.edu.cn"

def fc_op(input_op, name, n_out, layer_collector, act_func=tf.nn.leaky_relu):
    n_in = input_op.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        kernel = tf.Variable(tf.compat.v1.keras.initializers.glorot_normal()([n_in, n_out]), dtype=tf.float32, name=scope + "w")
        biases = tf.Variable(tf.constant(0, shape=[1, n_out], dtype=tf.float32), name=scope + 'b')
        fc = tf.add(tf.matmul(input_op, kernel), biases)
        activation = act_func(fc, name=scope + 'act')
        layer_collector.append([kernel, biases])
        return activation

class SDNE(object):
    def __init__(self, graph, encoder_layer_list, alpha=1e-6, beta=5., nu1=1e-5, nu2=1e-4,
                 batch_size=200, epoch=100, learning_rate=None):
        """
        encoder_layer_list: a list of numbers of the neuron at each encoder layer, the last number is the
        dimension of the output node representation
        Eg:
        if node size is 2000, encoder_layer_list=[1000, 128], then the whole neural network would be
        2000(input)->1000->128->1000->2000, SDNE extract the middle layer as the node representation
        """
        self.g = graph

        self.node_size = self.g.G.number_of_nodes()
        self.dim = encoder_layer_list[-1]

        self.encoder_layer_list = [self.node_size]
        self.encoder_layer_list.extend(encoder_layer_list)
        self.encoder_layer_num = len(encoder_layer_list) + 1

        self.alpha = alpha
        self.beta = beta
        self.nu1 = nu1
        self.nu2 = nu2
        self.bs = batch_size
        self.epoch = epoch
        self.max_iter = (epoch * self.node_size) // batch_size

        self.lr = learning_rate
        if self.lr is None:
            self.lr = tf.compat.v1.train.inverse_time_decay(0.03, self.max_iter, decay_steps=1, decay_rate=0.0001)

        tf.compat.v1.disable_eager_execution()
        self.sess = tf.compat.v1.Session()
        self.adj_mat = np.array(nx.adjacency_matrix(self.g.G).todense())
        self.embeddings = None

        self.build()
        
    def build(self):
        NodeA = tf.compat.v1.placeholder(tf.float32, shape=[None, self.node_size])
        NodeB = tf.compat.v1.placeholder(tf.float32, shape=[None, self.node_size])
        BmaskA = tf.compat.v1.placeholder(tf.float32, shape=[None, self.node_size])
        BmaskB = tf.compat.v1.placeholder(tf.float32, shape=[None, self.node_size])
        Weights = tf.compat.v1.placeholder(tf.float32, shape=[None])

        layer_collector = []
        nodes = tf.concat([NodeA, NodeB], axis=0)
        bmasks = tf.concat([BmaskA, BmaskB], axis=0)
        emb, recons = self.model(nodes, layer_collector, 'reconstructor')
        embs = tf.split(emb, num_or_size_splits=2, axis=0)

        L_1st = tf.reduce_sum(Weights * (tf.reduce_sum(tf.square(embs[0] - embs[1]), axis=1)))
        L_2nd = tf.reduce_sum(tf.square((nodes - recons) * bmasks))

        L = L_2nd + self.alpha * L_1st

        for param in layer_collector:
            L += self.nu1 * tf.reduce_sum(tf.abs(param[0])) + self.nu2 * tf.reduce_sum(tf.square(param[0]))

        optimizer = tf.compat.v1.train.AdamOptimizer(self.lr)
        train_op = optimizer.minimize(L)

        init = tf.compat.v1.global_variables_initializer()
        self.sess.run(init)

        generator = self.generate_batch()

        for step in range(self.max_iter + 1):
            nodes_a, nodes_b, beta_mask_a, beta_mask_b, weights = next(generator)

            feed_dict = {
                NodeA: nodes_a,
                NodeB: nodes_b,
                BmaskA: beta_mask_a,
                BmaskB: beta_mask_b,
                Weights: weights
            }

            self.sess.run(train_op, feed_dict=feed_dict)
            if step % 50 == 0:
                print("step %i: %s" % (step, self.sess.run([L, L_1st, L_2nd], feed_dict=feed_dict)))

        return self.sess.run(emb, feed_dict={NodeA: self.adj_mat[0:1, :], NodeB: self.adj_mat[1:, :]})

    def model(self, x, layer_collector, scope):
        with tf.compat.v1.variable_scope(scope, reuse=tf.compat.v1.AUTO_REUSE):
            h = x
            for i in range(1, self.encoder_layer_num):
                h = fc_op(h, 'encoder' + str(i), self.encoder_layer_list[i], layer_collector)
            for i in range(self.encoder_layer_num - 2, -1, -1):
                h = fc_op(h, 'decoder' + str(i), self.encoder_layer_list[i], layer_collector, act_func=tf.identity)
        return h, h

    def generate_batch(self):
        # Implement the batch generator method
        pass

    def get_embeddings(self):
        return self.embeddings

    def save_embeddings(self, filename):
        fout = open(filename, 'w')
        node_num = len(self.embeddings)
        fout.write("{} {}\n".format(node_num, self.dim))
        for node, vec in self.embeddings.items():
            fout.write("{} {}\n".format(node, ' '.join([str(x) for x in vec])))
        fout.close()
