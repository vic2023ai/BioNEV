# -*- coding: utf-8 -*-

import numpy as np
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize


class GraRep(object):

    def __init__(self, graph, Kstep, dim):
        self.g = graph
        self.Kstep = Kstep
        assert dim % Kstep == 0
        self.dim = int(dim / Kstep)
        self.train()

    def getAdjMat(self):
        graph = self.g.G
        node_size = self.g.node_size
        look_up = self.g.look_up_dict
        adj = np.zeros((node_size, node_size))
        for edge in self.g.G.edges():
            adj[look_up[edge[0]]][look_up[edge[1]]] = 1.0
            adj[look_up[edge[1]]][look_up[edge[0]]] = 1.0
        return np.matrix(adj)

    def GetProbTranMat(self, Ak):
        tileMat = np.tile(np.sum(Ak, axis=0), (self.node_size, 1))
        probTranMat = np.log(Ak / tileMat) - np.log(1.0 / self.node_size)
        probTranMat[probTranMat < 0] = 0
        probTranMat[np.isneginf(probTranMat)] = 0
        probTranMat[np.isnan(probTranMat)] = 0
        return probTranMat

    def GetRepUseSVD(self, probTranMat, alpha):
        k = min(self.dim, min(probTranMat.shape) - 1)
        if k <= 0:
            raise ValueError("El valor ajustado de 'k' no es válido. Verifique el tamaño del grafo y la dimensión deseada.")

        U, Sigma, VT = svds(probTranMat, k=k)
        Sigma = np.diag(Sigma)
        W = np.matmul(U, np.power(Sigma, alpha))
        C = np.matmul(VT.T, np.power(Sigma, alpha))
        embeddings = W + C
        return embeddings

    def save_embeddings(self, filename):
        fout = open(filename, 'w')
        node_num = len(self.vectors.keys())
        fout.write("{} {}\n".format(node_num, self.Kstep * self.dim))
        for node, vec in self.vectors.items():
            fout.write("{} {}\n".format(node, ' '.join([str(x) for x in vec])))
        fout.close()

    def train(self):
        self.adj = self.getAdjMat()
        self.node_size = self.adj.shape[0]
        self.Ak = np.matrix(np.identity(self.node_size))
        self.RepMat = np.zeros((self.node_size, int(self.dim * self.Kstep)))
        for i in range(self.Kstep):
            print('Kstep =', i)
            self.Ak = np.dot(self.Ak, self.adj)
            probTranMat = self.GetProbTranMat(self.Ak)
            Rk = self.GetRepUseSVD(probTranMat, 0.5)
            Rk = normalize(Rk, axis=1, norm='l2')
            self.RepMat[:, self.dim * i:self.dim * (i + 1)] = Rk[:, :]
        self.vectors = {}
        look_back = self.g.look_back_list
        for i, embedding in enumerate(self.RepMat):
            self.vectors[look_back[i]] = embedding
