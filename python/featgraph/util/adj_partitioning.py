import scipy
import scipy.sparse
import numpy as np
import copy


def util_convert_csr_to_dds(adj_scipy_csr, num_col_partitions):
    """Doing src vertex partitioning (column dimension) by converting csr to dds (dense-dense-sparse).

    Parameters
    ----------
    adj_scipy_csr : scipy.sparse.csr_matrix
        The input matrix to be partitioned

    num_col_partitions : int
        Number of partitions along the column dimension

    Returns
    -------
    s1_pos : numpy.array, dtype is int32
        1-D with shape [num_col_partitions * (num_rows + 1)]

    s1_idx : numpy.array, dtype is int32
        1-D with shape [nnz]

    vals : numpy.array, dtype is float32
        1-D with shape [nnz]
    """
    num_rows = adj_scipy_csr.shape[0]
    num_cols = adj_scipy_csr.shape[1]
    adj_data = adj_scipy_csr.data
    adj_indices = adj_scipy_csr.indices
    adj_indptr = adj_scipy_csr.indptr
    d1_size = num_col_partitions
    d2_size = adj_indptr.shape[0]

    s1_pos = np.zeros(shape=(d1_size*d2_size), dtype=adj_indptr.dtype)
    s1_idx = np.zeros(shape=adj_indices.shape, dtype=adj_indices.dtype)
    vals = np.zeros(shape=adj_data.shape, dtype=adj_data.dtype)

    counter = 0
    num_cols_per_partition = (num_cols + num_col_partitions - 1) // num_col_partitions
    for i in range(num_col_partitions):
        if i == num_col_partitions - 1:
            adj_partition_scipy_csr = adj_scipy_csr[:, (i*num_cols_per_partition)::]
        else:
            adj_partition_scipy_csr = adj_scipy_csr[:, (i*num_cols_per_partition):(i+1)*num_cols_per_partition]
        nnz = adj_partition_scipy_csr.data.shape[0]
        vals[counter:(counter+nnz)] = adj_partition_scipy_csr.data
        s1_pos[i*d2_size:(i+1)*d2_size] = adj_partition_scipy_csr.indptr + counter
        s1_idx[counter:(counter+nnz)] = adj_partition_scipy_csr.indices
        counter += nnz

    assert len(s1_idx) == counter
    assert len(vals) == counter

    return s1_pos, s1_idx, vals


def util_partition_adj_coo_2d(adj_scipy_coo, num_row_partitions, num_col_partitions):
    """Doing 2D partitioning.

    Parameters
    ----------
    adj_scipy_coo : scipy.sparse.coo_matrix
        The input matrix to be partitioned

    num_row_partitions : int
        Number of partitions along the row dimension

    num_col_partitions : int
        Number of partitions along the col dimension

    Returns
    -------
    edge_id_list_after_partition : numpy.array, dtype is int32
        1-D with shape [nnz] (COO)

    adj_row_indices_after_partition : numpy.array, dtype is int32
        1-D with shape [nnz] (COO)

    adj_col_indices_after_partition : numpy.array, dtype is int32
        1-D with shape [nnz] (COO)
    """
    num_rows = adj_scipy_coo.shape[0]
    num_cols = adj_scipy_coo.shape[1]
    adj_row_indices = adj_scipy_coo.row
    adj_col_indices = adj_scipy_coo.col
    nnz = adj_row_indices.shape[0]
    assert adj_col_indices.shape[0] == nnz, "length of adj_row_indices and that of adj_col_indices do not match"
    # Give each edge an id to record the graph traversal order
    adj_scipy_coo.data = np.arange(1, 1 + nnz, dtype='int32')

    # COO matrix is not subscriptable; we need CSR to do partitioning
    adj_scipy_csr = adj_scipy_coo.tocsr()
    edge_id_list_after_partition = np.zeros(shape=(nnz), dtype=adj_scipy_coo.data.dtype)
    adj_row_indices_after_partition = np.zeros(shape=(nnz), dtype=adj_row_indices.dtype)
    adj_col_indices_after_partition = np.zeros(shape=(nnz), dtype=adj_col_indices.dtype)

    num_rows_per_partition = (num_rows + num_row_partitions - 1) // num_row_partitions
    num_cols_per_partition = (num_cols + num_col_partitions - 1) // num_col_partitions
    counter = 0
    for row_idx in range(num_row_partitions):
        for col_idx in range(num_col_partitions):
            if row_idx < num_row_partitions - 1 and col_idx < num_col_partitions - 1:
                adj_partition_scipy_csr = adj_scipy_csr[row_idx*num_rows_per_partition:(row_idx+1)*num_rows_per_partition, \
                    col_idx*num_cols_per_partition:(col_idx+1)*num_cols_per_partition]
            elif row_idx < num_row_partitions - 1 and col_idx == num_col_partitions - 1:
                adj_partition_scipy_csr = adj_scipy_csr[row_idx*num_rows_per_partition:(row_idx+1)*num_rows_per_partition, \
                    col_idx*num_cols_per_partition::]
            elif row_idx == num_row_partitions - 1 and col_idx < num_col_partitions - 1:
                adj_partition_scipy_csr = adj_scipy_csr[row_idx*num_rows_per_partition::, \
                    col_idx*num_cols_per_partition:(col_idx+1)*num_cols_per_partition]
            elif row_idx == num_row_partitions - 1 and col_idx == num_col_partitions - 1:
                adj_partition_scipy_csr = adj_scipy_csr[row_idx*num_rows_per_partition::, \
                    col_idx*num_cols_per_partition::]
            else:
                raise RuntimeError("no condition is satisfied")
            adj_partition_scipy_coo = adj_partition_scipy_csr.tocoo()
            nnz_in_this_partition = adj_partition_scipy_coo.nnz
            edge_id_list_after_partition[counter:(counter+nnz_in_this_partition)] = adj_partition_scipy_coo.data
            adj_row_indices_after_partition[counter:(counter+nnz_in_this_partition)] = \
                adj_partition_scipy_coo.row + row_idx * num_rows_per_partition
            adj_col_indices_after_partition[counter:(counter+nnz_in_this_partition)] = \
                adj_partition_scipy_coo.col + col_idx * num_cols_per_partition
            counter += nnz_in_this_partition

    return edge_id_list_after_partition, adj_row_indices_after_partition, adj_col_indices_after_partition
