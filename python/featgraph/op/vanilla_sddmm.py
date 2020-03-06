import tvm
from topi.util import get_const_tuple


def vanilla_sddmm(SrcFeat,
                  DstFeat,
                  Adj_row_indices,
                  Adj_col_indices,
                  num_feat_partitions=1):
    # TODO：apply parallelization in cpu schedule
    # TODO: support tuning both block number and thread number in cuda schedule
    """Compute sampled dense dense matrix multiplication of SrcFeat and DstFeat with Adj matrix as mask.

    Parameters
    ----------
    SrcFeat : tvm.Tensor
        2-D with shape [num_rows, feat_len]

    DstFeat : tvm.Tensor
        2-D with shape [num_cols, feat_len]

    Adj_row_indices : tvm.Tensor
        1-D with shape [nnz] (COO)

    Adj_col_indices : tvm.Tensor
        1-D with shape [nnz] (COO)

    num_feat_partitions : int
        Doing feature dimension tiling

    Returns
    -------
    Out : tvm.Tensor
        1-D with shape [nnz] (COO)
    """
    feat_len = get_const_tuple(SrcFeat.shape)[1]
    assert get_const_tuple(DstFeat.shape)[1] == feat_len, "dimension mismatch"
    num_edges = get_const_tuple(Adj_row_indices.shape)[0]
    assert get_const_tuple(Adj_col_indices.shape)[0] == num_edges, "dimension mismatch"
    oshape = (num_edges,)

    k = tvm.reduce_axis((0, feat_len))

    if num_feat_partitions == 1:
        def edgefunc(eid):  # eid: edge id
            return tvm.sum(SrcFeat[Adj_row_indices[eid], k] * DstFeat[Adj_col_indices[eid], k], axis=k)
    else:
        feat_len_per_partition = feat_len // num_feat_partitions  # we assume feat_len % num_feat_partitions = 0
        num_rows = get_const_tuple(SrcFeat.shape)[0]
        num_cols = get_const_tuple(DstFeat.shape)[0]
        ReshapedSrcFeat = tvm.compute((num_feat_partitions, num_rows, feat_len_per_partition), \
                                       lambda fo, nn, fi: SrcFeat[nn, fo*feat_len_per_partition + fi], \
                                       name='ReshapedSrcFeat')
        ReshapedDstFeat = tvm.compute((num_feat_partitions, num_cols, feat_len_per_partition), \
                                       lambda fo, nn, fi: DstFeat[nn, fo*feat_len_per_partition + fi], \
                                       name='ReshapedDstFeat')
        def edgefunc(eid):  # eid: edge id
            return tvm.sum(ReshapedSrcFeat[k // feat_len_per_partition, Adj_row_indices[eid], k % feat_len_per_partition] \
                           * ReshapedDstFeat[k // feat_len_per_partition, Adj_col_indices[eid], k % feat_len_per_partition], axis=k)

    Out = tvm.compute(oshape, edgefunc, name='vanilla_sddmm')
    return Out


def schedule_vanilla_sddmm_x86(Out, num_feat_partitions=1):
    s = tvm.create_schedule([Out.op])
    if num_feat_partitions != 1:
        edge_iter_axis = Out.op.axis[0]
        feat_reduce_axis = Out.op.reduce_axis[0]
        fo, fi = s[Out.op].split(feat_reduce_axis, nparts=num_feat_partitions)
        s[Out.op].reorder(fo, edge_iter_axis, fi)
    return s


def schedule_vanilla_sddmm_cuda_tree_reduce(Out, num_feat_partitions=1, num_cuda_blocks=8192):
    s = tvm.create_schedule([Out.op])
    assert num_feat_partitions == 1, "cuda schedule for sddmm does not support feat dimension tiling, " \
                                     "which requires cross-cuda-block reduction and atomic operations."
    edge_iter_axis = Out.op.axis[0]
    block_idx, _ = s[Out.op].split(edge_iter_axis, nparts=num_cuda_blocks)
    s[Out.op].bind(block_idx, tvm.thread_axis("blockIdx.x"))
    # pay attention: here is doing tree reduce
    s[Out.op].bind(Out.op.reduce_axis[0], tvm.thread_axis("threadIdx.x"))
    return s


def schedule_vanilla_sddmm_cuda_single_thread_reduce(Out, num_feat_partitions=1, num_threads_per_cuda_block=32):
    s = tvm.create_schedule([Out.op])
    assert num_feat_partitions == 1, "cuda schedule for sddmm does not support feat dimension tiling, " \
                                     "which requires cross-cuda-block reduction and atomic operations."
    edge_iter_axis = Out.op.axis[0]
    block_idx, thread_idx = s[Out.op].split(edge_iter_axis, factor=num_threads_per_cuda_block)
    s[Out.op].bind(block_idx, tvm.thread_axis("blockIdx.x"))
    s[Out.op].bind(thread_idx, tvm.thread_axis("threadIdx.x"))
    return s
