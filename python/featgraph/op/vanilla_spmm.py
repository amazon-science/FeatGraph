import tvm
from topi.util import get_const_tuple


def vanilla_spmm_csr_x86(SrcFeat,
                         Adj_indptr,
                         Adj_indices,
                         Adj_vals,
                         num_feat_partitions=1):
    """Compute sparse-dense matrix multiplication of Adj and SrcFeat on x86.
    This implementation applies feature dimension partitioning, which requires transforming the layout of SrcFeat.

    Parameters
    ----------
    SrcFeat : tvm.Tensor
        2-D with shape [num_src_vertices, feat_len]

    Adj_indptr : tvm.Tensor
        1-D with shape [num_dst_vertices + 1] (CSR)

    Adj_indices : tvm.Tensor
        1-D with shape [nnz] (CSR)

    Adj_vals : tvm.Tensor
        1-D with shape [nnz] (CSR)

    num_feat_partitions : int
        Doing feature dimension tiling

    Returns
    -------
    Out : tvm.Tensor
        2-D with shape [num_dst_vertices, feat_len]
    """
    assert Adj_indices.shape[0].value == Adj_vals.shape[0].value
    num_src_vertices, feat_len = get_const_tuple(SrcFeat.shape)
    num_dst_vertices = Adj_indptr.shape[0].value - 1
    oshape = (num_dst_vertices, feat_len)

    feat_len_per_partition = feat_len // num_feat_partitions  # we assume feat_len % num_feat_partitions = 0

    ReshapedSrcFeat = tvm.compute((num_feat_partitions, num_src_vertices, feat_len_per_partition), \
        lambda fo, nn, fi: SrcFeat[nn, fo * feat_len_per_partition + fi], name='ReshapedSrcFeat')

    def msgfunc(fo, row, fi):
        row_start = Adj_indptr[row]
        row_end = Adj_indptr[row + 1]
        row_num_elems = row_end - row_start
        elem_idx = tvm.reduce_axis((0, row_num_elems), name="elem_idx")
        adj_val = Adj_vals[row_start + elem_idx]
        feat_val = ReshapedSrcFeat[fo, Adj_indices[row_start + elem_idx], fi]
        return tvm.sum(adj_val * feat_val, axis=elem_idx)

    ReshapedOut = tvm.compute((num_feat_partitions, num_dst_vertices, feat_len_per_partition),
        msgfunc, name='ReshapedOut')

    Out = tvm.compute(oshape, \
        lambda nn, ff: ReshapedOut[ff // feat_len_per_partition, nn, ff % feat_len_per_partition], \
        name='Out')

    return Out


def schedule_vanilla_spmm_csr_x86(Out):
    s = tvm.create_schedule([Out.op])

    ReshapedOut = Out.op.input_tensors[0]
    ReshapedSrcFeat = ReshapedOut.op.input_tensors[3]

    # Reorder
    RO = ReshapedOut
    s[RO.op].reorder(RO.op.axis[0], RO.op.axis[1], RO.op.reduce_axis[0], RO.op.axis[2])

    # Parallelize the rows of the sparse matrix
    s[ReshapedSrcFeat.op].parallel(ReshapedSrcFeat.op.axis[1])
    s[ReshapedOut.op].parallel(ReshapedOut.op.axis[1])
    s[Out.op].parallel(Out.op.axis[0])

    return s


def vanilla_spmm_csr_cuda(SrcFeat,
                          Adj_indptr,
                          Adj_indices,
                          Adj_vals):
    """Compute sparse-dense matrix multiplication of Adj and SrcFeat on cuda.
    This implementation does not transform the layout of SrcFeat.

    Parameters
    ----------
    SrcFeat : tvm.Tensor
        2-D with shape [num_src_vertices, feat_len]

    Adj_indptr : tvm.Tensor
        1-D with shape [num_dst_vertices + 1] (CSR)

    Adj_indices : tvm.Tensor
        1-D with shape [nnz] (CSR)

    Adj_vals : tvm.Tensor
        1-D with shape [nnz] (CSR)

    Returns
    -------
    Out : tvm.Tensor
        2-D with shape [num_dst_vertices, feat_len]
    """
    assert Adj_indices.shape[0].value == Adj_vals.shape[0].value
    num_src_vertices, feat_len = get_const_tuple(SrcFeat.shape)
    num_dst_vertices = Adj_indptr.shape[0].value - 1
    oshape = (num_dst_vertices, feat_len)

    def msgfunc(row, ff):
        row_start = Adj_indptr[row]
        row_end = Adj_indptr[row + 1]
        row_num_elems = row_end - row_start
        elem_idx = tvm.reduce_axis((0, row_num_elems), name="elem_idx")
        adj_val = Adj_vals[row_start + elem_idx]
        feat_val = SrcFeat[Adj_indices[row_start + elem_idx], ff]
        return tvm.sum(adj_val * feat_val, axis=elem_idx)

    Out = tvm.compute(oshape, msgfunc, name='Out')

    return Out


def schedule_vanilla_spmm_csr_cuda(Out,
                                   num_cuda_blocks=None,
                                   num_threads_per_cuda_block=None):
    s = tvm.create_schedule([Out.op])
    num_rows = Out.shape[0].value
    feat_len = Out.shape[1].value
    if num_cuda_blocks is None:
        num_cuda_blocks = num_rows
    if num_threads_per_cuda_block is None:
        num_threads_per_cuda_block = feat_len
    row_axis = Out.op.axis[0]
    feat_axis = Out.op.axis[1]
    row_outer, row_inner = s[Out.op].split(row_axis, nparts=num_cuda_blocks)
    feat_outer, feat_inner = s[Out.op].split(feat_axis, factor=num_threads_per_cuda_block)
    s[Out.op].reorder(feat_outer, row_outer, feat_inner, row_inner)
    s[Out.op].bind(feat_outer, tvm.thread_axis("blockIdx.y"))
    s[Out.op].bind(row_outer, tvm.thread_axis("blockIdx.x"))
    s[Out.op].bind(feat_inner, tvm.thread_axis("threadIdx.x"))
    return s


def vanilla_spmm_dds_x86(SrcFeat,
                         Adj_s1_pos,
                         Adj_s1_idx,
                         Adj_vals,
                         d1_size,
                         d2_size,
                         num_feat_partitions=1):
    """Compute sparse-dense matrix multiplication of Adj and SrcFeat on x86.
    This implementation applies both feature dimension partitioning and 1D graph partitioning.
    1D graph partitioning transforms the csr Adj matrix into dense-dense-sparse (DDS) format.

    Parameters
    ----------
    SrcFeat : tvm.Tensor
        2-D with shape [num_src_vertices, feat_len]

    Adj_s1_pos : tvm.Tensor
        1-D with shape [d1_size * d2_size] (DDS)

    Adj_s1_idx : tvm.Tensor
        1-D with shape [nnz] (DDS)

    Adj_vals : tvm.Tensor
        1-D with shape [nnz] (DDS)

    d1_size : int
        Number of src vertex partitions

    d2_size : int
        num_dst_vertices + 1

    num_feat_partitions : int
        Doing feature dimension tiling

    Returns
    -------
    Out : tvm.Tensor
        2-D with shape [num_dst_vertices, feat_len]
    """
    assert d1_size * d2_size == Adj_s1_pos.shape[0].value
    assert Adj_s1_idx.shape[0].value == Adj_vals.shape[0].value
    num_src_vertices, feat_len = get_const_tuple(SrcFeat.shape)
    num_src_vertex_partitions = d1_size
    num_dst_vertices = d2_size - 1
    oshape = (num_dst_vertices, feat_len)

    feat_len_per_partition = feat_len // num_feat_partitions  # we assume feat_len % num_feat_partitions = 0
    num_src_vertices_per_partition = (num_src_vertices + num_src_vertex_partitions - 1) // num_src_vertex_partitions

    ReshapedSrcFeat = tvm.compute((num_feat_partitions, num_src_vertices, feat_len_per_partition), \
        lambda fo, nn, fi: SrcFeat[nn, fo * feat_len_per_partition + fi], name='ReshapedSrcFeat')

    def msgfunc(fo, src_vertex_partition_idx, row, fi):
        row_start = Adj_s1_pos[src_vertex_partition_idx * d2_size + row]
        row_end = Adj_s1_pos[src_vertex_partition_idx * d2_size + row + 1]
        row_num_elems = row_end - row_start
        elem_idx = tvm.reduce_axis((0, row_num_elems), name="elem_idx")
        adj_val = Adj_vals[row_start + elem_idx]
        feat_val = ReshapedSrcFeat[fo, \
                                   Adj_s1_idx[row_start + elem_idx] + src_vertex_partition_idx * num_src_vertices_per_partition, \
                                   fi]
        return tvm.sum(adj_val * feat_val, axis=elem_idx)

    Intermediate = tvm.compute((num_feat_partitions, num_src_vertex_partitions, num_dst_vertices, feat_len_per_partition), \
        msgfunc, name='Intermediate')

    k = tvm.reduce_axis((0, num_src_vertex_partitions), name='src_vertex_partition_reduce')
    ReshapedOut = tvm.compute((num_feat_partitions, num_dst_vertices, feat_len_per_partition),
        lambda fo, nn, fi: tvm.sum(Intermediate[fo, k, nn, fi], axis=k), \
        name='ReshapedOut')

    Out = tvm.compute(oshape, \
        lambda nn, ff: ReshapedOut[ff // feat_len_per_partition, nn, ff % feat_len_per_partition], \
        name='Out')

    return Out


def schedule_vanilla_spmm_dds_x86(Out):
    s = tvm.create_schedule([Out.op])

    ReshapedOut = Out.op.input_tensors[0]
    Intermediate = ReshapedOut.op.input_tensors[0]
    ReshapedSrcFeat = Intermediate.op.input_tensors[3]

    I = Intermediate
    RO = ReshapedOut
    s[I.op].reorder(I.op.axis[0], I.op.axis[1], I.op.axis[2], I.op.reduce_axis[0], I.op.axis[3])
    s[RO.op].reorder(RO.op.axis[0], RO.op.reduce_axis[0], RO.op.axis[1], RO.op.axis[2])
    s[I.op].compute_at(s[RO], RO.op.reduce_axis[0])

    # Parallelize the rows of the sparse matrix
    s[ReshapedSrcFeat.op].parallel(ReshapedSrcFeat.op.axis[1])
    s[Intermediate.op].parallel(Intermediate.op.axis[2])
    s[ReshapedOut.op].parallel(ReshapedOut.op.axis[1])
    s[Out.op].parallel(Out.op.axis[0])

    return s
