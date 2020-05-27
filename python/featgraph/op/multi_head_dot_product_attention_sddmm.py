import tvm
from topi.util import get_const_tuple


def multi_head_dot_product_attention_sddmm(SrcFeat,
                                           DstFeat,
                                           Adj_row_indices,
                                           Adj_col_indices,
                                           num_head_partitions=1,
                                           num_feat_partitions=1):
    """Compute multi-head dot-product attention of SrcFeat and DstFeat with Adj matrix as mask.

    Parameters
    ----------
    SrcFeat : tvm.Tensor
        3-D with shape [num_rows, num_heads, feat_len]

    DstFeat : tvm.Tensor
        3-D with shape [num_cols, num_heads, feat_len]

    Adj_row_indices : tvm.Tensor
        1-D with shape [nnz] (COO)

    Adj_col_indices : tvm.Tensor
        1-D with shape [nnz] (COO)

    num_head_partitions : int
        Doing head dimension tiling

    num_feat_partitions : int
        Doing feature dimension tiling

    Returns
    -------
    Out : tvm.Tensor
        2-D with shape [nnz, num_heads] (COO)
    """
    _, num_heads, feat_len = get_const_tuple(SrcFeat.shape)
    assert get_const_tuple(DstFeat.shape)[1] == num_heads, "dimension mismatch"
    assert get_const_tuple(DstFeat.shape)[2] == feat_len, "dimension mismatch"
    num_edges = get_const_tuple(Adj_row_indices.shape)[0]
    assert get_const_tuple(Adj_col_indices.shape)[0] == num_edges, "dimension mismatch"
    oshape = (num_edges, num_heads)

    k = tvm.reduce_axis((0, feat_len))

    if num_feat_partitions == 1 and num_head_partitions == 1:
        def edgefunc(eid, hid):  # eid: edge id, hid: head id
            return tvm.sum(SrcFeat[Adj_col_indices[eid], hid, k] * DstFeat[Adj_row_indices[eid], hid, k], axis=k)
    else:
        num_heads_per_partition = num_heads // num_head_partitions  # we assume num_heads % num_head_partitions = 0
        feat_len_per_partition = feat_len // num_feat_partitions  # we assume feat_len % num_feat_partitions = 0
        num_rows = get_const_tuple(SrcFeat.shape)[0]
        num_cols = get_const_tuple(DstFeat.shape)[0]
        ReshapedSrcFeat = tvm.compute((num_head_partitions, num_feat_partitions, num_rows, num_heads_per_partition, feat_len_per_partition), \
                                       lambda ho, fo, nn, hi, fi: SrcFeat[nn, ho*num_heads_per_partition + hi, fo*feat_len_per_partition + fi], \
                                       name='ReshapedSrcFeat')
        ReshapedDstFeat = tvm.compute((num_head_partitions, num_feat_partitions, num_cols, num_heads_per_partition, feat_len_per_partition), \
                                       lambda ho, fo, nn, hi, fi: DstFeat[nn, ho*num_heads_per_partition + hi, fo*feat_len_per_partition + fi], \
                                       name='ReshapedDstFeat')
        # TODO: also transform the layout of Out to improve write locality?
        def edgefunc(eid, hid):  # eid: edge id, hid: head id
            return tvm.sum(ReshapedSrcFeat[hid // num_heads_per_partition, \
                                           k // feat_len_per_partition, \
                                           Adj_col_indices[eid], \
                                           hid % num_heads_per_partition, \
                                           k % feat_len_per_partition] \
                           * ReshapedDstFeat[hid // num_heads_per_partition, \
                                             k // feat_len_per_partition, \
                                             Adj_row_indices[eid], \
                                             hid % num_heads_per_partition, \
                                             k % feat_len_per_partition], axis=k)

    Out = tvm.compute(oshape, edgefunc, name='multi_head_dot_product_attention_sddmm')
    return Out


def schedule_multi_head_dot_product_attention_sddmm_x86(Out, num_head_partitions=1, num_feat_partitions=1):
    s = tvm.create_schedule([Out.op])
    if num_feat_partitions != 1 or num_head_partitions != 1:
        edge_iter_axis = Out.op.axis[0]
        head_iter_axis = Out.op.axis[1]
        ho, hi = s[Out.op].split(head_iter_axis, nparts=num_head_partitions)
        feat_reduce_axis = Out.op.reduce_axis[0]
        fo, fi = s[Out.op].split(feat_reduce_axis, nparts=num_feat_partitions)
        s[Out.op].reorder(ho, fo, edge_iter_axis, hi, fi)
    return s
