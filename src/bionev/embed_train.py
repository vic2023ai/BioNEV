# -*- coding: utf-8 -*-

import ast
import logging
import os

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

from bionev.GAE.train_model import gae_model
from bionev.OpenNE import node2vec
from bionev.SVD.model import SVD_embedding
from bionev.struc2vec import struc2vec
from bionev.utils import *


def embedding_training(args, train_graph_filename):
    if args.method == 'struc2vec':
        g = read_for_struc2vec(train_graph_filename)
    elif args.method == 'GAE':
        g = read_for_gae(train_graph_filename)
    elif args.method == 'SVD':
        g = read_for_SVD(train_graph_filename, weighted=args.weighted)
    else:
        g = read_for_OpenNE(train_graph_filename, weighted=args.weighted)

    _embedding_training(args, G_=g)

    return


def _embedding_training(args, G_=None):
    seed=args.seed

    if args.method == 'struc2vec':
        logging.basicConfig(filename='./src/bionev/struc2vec/struc2vec.log', filemode='w', level=logging.DEBUG,
                            format='%(asctime)s %(message)s')
        if (args.OPT3):
            until_layer = args.until_layer
        else:
            until_layer = None

        G = struc2vec.Graph(G_, args.workers, untilLayer=until_layer)

        if (args.OPT1):
            G.preprocess_neighbors_with_bfs_compact()
        else:
            G.preprocess_neighbors_with_bfs()

        if (args.OPT2):
            G.create_vectors()
            G.calc_distances(compactDegree=args.OPT1)
        else:
            G.calc_distances_all_vertices(compactDegree=args.OPT1)

        print('create distances network..')
        G.create_distances_network()
        print('begin random walk...')
        G.preprocess_parameters_random_walk()

        G.simulate_walks(args.number_walks, args.walk_length)
        print('walk finished..\nLearning embeddings...')
        walks = LineSentence('random_walks.txt')
        model = Word2Vec(walks, vector_size=args.dimensions, window=args.window_size, min_count=0, hs=1, sg=1,
                         workers=args.workers, seed=seed)
        os.remove("random_walks.txt")
        model.wv.save_word2vec_format(args.output)
    elif args.method == 'GAE':
        model = gae_model(args)
        G = G_[0]
        node_list = G_[1]
        model.train(G)
        # save embeddings
        model.save_embeddings(args.output, node_list)
    elif args.method == 'SVD':
        SVD_embedding(G_, args.output, size=args.dimensions)
    else:

        if args.method == 'node2vec':
            model = node2vec.Node2vec(graph=G_, path_length=args.walk_length,
                                      num_paths=args.number_walks, dim=args.dimensions,
                                      workers=args.workers, p=args.p, q=args.q, window=args.window_size)

        else:
            raise ValueError(f'Invalid method: {args.method}')

        print("Saving embeddings...")
        model.save_embeddings(args.output)

    return
