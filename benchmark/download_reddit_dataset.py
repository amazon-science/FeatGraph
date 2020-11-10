import os
import scipy.sparse
import dgl
from dgl.data import RedditDataset

if not os.path.isdir('data'):
    os.mkdir('data')

data = RedditDataset()

adj_scipy_csr = data.graph.adjacency_matrix_scipy(fmt='csr')
adj_scipy_csr.data = adj_scipy_csr.data.astype('float32')
scipy.sparse.save_npz('data/reddit_csr_float32.npz', adj_scipy_csr)

adj_scipy_coo = data.graph.adjacency_matrix_scipy(fmt='coo')
adj_scipy_coo.data = adj_scipy_coo.data.astype('float32')
scipy.sparse.save_npz('data/reddit_coo_float32.npz', adj_scipy_coo)
