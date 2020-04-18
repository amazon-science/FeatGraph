import scipy
import scipy.sparse
import numpy as np

from featgraph.util import util_convert_csr_to_dds, util_partition_adj_coo_2d


def test_util_convert_csr_to_dds():
    row  = np.array([0, 0, 0, 0, 1, 1, 2, 3, 4])
    col  = np.array([0, 1, 2, 3, 0, 2, 1, 3, 2])
    edge_id_list = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    sample_coo_matrix = scipy.sparse.coo_matrix((edge_id_list, (row, col)), shape=(5, 5), dtype='int32')
    sample_coo_matrix_dense_view = sample_coo_matrix.toarray()
    A = np.array([[1, 2, 3, 4, 0],
                  [5, 0, 6, 0, 0],
                  [0, 7, 0, 0, 0],
                  [0, 0, 0, 8, 0],
                  [0, 0, 9, 0, 0]])
    np.testing.assert_allclose(A, sample_coo_matrix_dense_view)
    adj_col_num_partition = 2
    s1_pos, s1_idx, vals = util_convert_csr_to_dds(sample_coo_matrix.tocsr(), adj_col_num_partition)
    np.testing.assert_allclose(vals, np.array([1, 2, 3, 5, 6, 7, 9, 4, 8]))
    np.testing.assert_allclose(s1_idx, np.array([0, 1, 2, 0, 2, 1, 2, 0, 0]))
    np.testing.assert_allclose(s1_pos, np.array([0, 3, 5, 6, 6, 7, 7, 8, 8, 8, 9, 9]))


def test_util_partition_adj_coo_2d():
    row  = np.array([0, 0, 0, 0, 1, 1, 2, 3, 4])
    col  = np.array([0, 1, 2, 3, 0, 2, 1, 3, 2])
    edge_id_list = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    sample_coo_matrix = scipy.sparse.coo_matrix((edge_id_list, (row, col)), shape=(5, 5), dtype='int32')
    sample_coo_matrix_dense_view = sample_coo_matrix.toarray()
    A = np.array([[1, 2, 3, 4, 0],
                  [5, 0, 6, 0, 0],
                  [0, 7, 0, 0, 0],
                  [0, 0, 0, 8, 0],
                  [0, 0, 9, 0, 0]])
    np.testing.assert_allclose(A, sample_coo_matrix_dense_view)
    adj_row_num_partition = 2
    adj_col_num_partition = 2
    edge_id_list_after_partition, adj_row_indices_after_partition, adj_col_indices_after_partition = \
        util_partition_adj_coo_2d(sample_coo_matrix, adj_row_num_partition, adj_col_num_partition)
    np.testing.assert_allclose(edge_id_list_after_partition, np.array([1, 2, 3, 5, 6, 7, 4, 9, 8]))


if __name__ == '__main__':
    test_util_convert_csr_to_dds()
    test_util_partition_adj_coo_2d()
