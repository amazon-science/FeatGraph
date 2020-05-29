import scipy
import scipy.sparse
import numpy as np
import time
import tvm
from topi.util import get_const_tuple

from featgraph.module import VanillaSDDMMx86, VanillaSDDMMcuda, MultiHeadSDDMMx86, MultiHeadSDDMMcuda


def test_vanilla_sddmm(adj_scipy_coo, target):
    num_rows = adj_scipy_coo.shape[0]
    num_cols = adj_scipy_coo.shape[1]

    # doing 2D graph partitioning during initialization
    # note that 2D partitioning is mainly useful for CPU since it optimizes cache
    if target == 'x86':
        num_row_partitions = 4
        num_col_partitions = 4
        vanilla_sddmm_module = VanillaSDDMMx86(adj_scipy_coo, num_row_partitions, num_col_partitions)
    elif target == 'cuda':
        vanilla_sddmm_module = VanillaSDDMMcuda(adj_scipy_coo)

    # tvm func is built for a specific feat_len and num_feat_partitions
    feat_len = 128
    SrcFeat = tvm.placeholder((num_cols, feat_len))
    DstFeat = tvm.placeholder((num_rows, feat_len))
    input_placeholders = [SrcFeat, DstFeat]
    if target == 'x86':
        num_feat_partitions = 4
        compute_args = {'num_feat_partitions': num_feat_partitions}
        schedule_args = {'num_feat_partitions': num_feat_partitions}
    elif target == 'cuda':
        num_cuda_blocks = 4096
        num_threads_per_cuda_block = 64
        compute_args = {}
        schedule_args = {'num_cuda_blocks': num_cuda_blocks}
    vanilla_sddmm_module.build(input_placeholders, compute_args, schedule_args)

    # run
    src_feat_np = np.random.random(get_const_tuple(SrcFeat.shape)).astype('float32')
    dst_feat_np = np.random.random(get_const_tuple(DstFeat.shape)).astype('float32')
    src_feat_tvm = tvm.nd.array(src_feat_np, vanilla_sddmm_module.ctx)
    dst_feat_tvm = tvm.nd.array(dst_feat_np, vanilla_sddmm_module.ctx)
    input_tvm_ndarrays = [src_feat_tvm, dst_feat_tvm]
    out_tvm = vanilla_sddmm_module.run(input_tvm_ndarrays).asnumpy()
    # be careful here
    if target == 'x86':
        out_tvm = out_tvm[vanilla_sddmm_module.edge_mapping]

    # check correctness against scipy
    lhs = src_feat_np[adj_scipy_coo.col]
    rhs = dst_feat_np[adj_scipy_coo.row]
    out_scipy = (lhs * rhs).sum(axis=-1)
    np.testing.assert_allclose(out_scipy, out_tvm, rtol=1e-4, atol=1e-4)


def test_multi_head_dot_product_attention_sddmm(adj_scipy_coo, target):
    num_rows = adj_scipy_coo.shape[0]
    num_cols = adj_scipy_coo.shape[1]

    # doing 2D graph partitioning during initialization
    # note that 2D partitioning is mainly useful for CPU since it optimizes cache
    if target == 'x86':
        num_row_partitions = 4
        num_col_partitions = 4
        multi_head_sddmm_module = MultiHeadSDDMMx86(adj_scipy_coo, num_row_partitions, num_col_partitions)
    elif target == 'cuda':
        multi_head_sddmm_module = MultiHeadSDDMMcuda(adj_scipy_coo)

    # tvm func is built for a specific num_heads, num_head_partitions, feat_len, num_feat_partitions
    num_heads = 16
    feat_len = 64
    SrcFeat = tvm.placeholder((num_cols, num_heads, feat_len))
    DstFeat = tvm.placeholder((num_rows, num_heads, feat_len))
    input_placeholders = [SrcFeat, DstFeat]
    if target == 'x86':
        num_head_partitions = 2
        num_feat_partitions = 8
        compute_args = {'num_head_partitions': num_head_partitions,
                        'num_feat_partitions': num_feat_partitions}
        schedule_args = {'num_head_partitions': num_head_partitions,
                         'num_feat_partitions': num_feat_partitions}
    elif target == 'cuda':
        num_cuda_blocks = 4096
        num_threads_per_cuda_block = 64
        compute_args = {}
        schedule_args = {'num_cuda_blocks': num_cuda_blocks,
                         'num_threads_per_cuda_block': num_threads_per_cuda_block}
    multi_head_sddmm_module.build(input_placeholders, compute_args, schedule_args)

    # run
    src_feat_np = np.random.random(get_const_tuple(SrcFeat.shape)).astype('float32')
    dst_feat_np = np.random.random(get_const_tuple(DstFeat.shape)).astype('float32')
    src_feat_tvm = tvm.nd.array(src_feat_np, multi_head_sddmm_module.ctx)
    dst_feat_tvm = tvm.nd.array(dst_feat_np, multi_head_sddmm_module.ctx)
    input_tvm_ndarrays = [src_feat_tvm, dst_feat_tvm]
    out_tvm = multi_head_sddmm_module.run(input_tvm_ndarrays).asnumpy()
    # be careful here
    if target == 'x86':
        out_tvm = out_tvm[multi_head_sddmm_module.edge_mapping]

    # check correctness against scipy
    lhs = src_feat_np[adj_scipy_coo.col]
    rhs = dst_feat_np[adj_scipy_coo.row]
    out_scipy = (lhs * rhs).sum(axis=-1)
    np.testing.assert_allclose(out_scipy, out_tvm, rtol=1e-4, atol=1e-4)


if __name__ == '__main__':
    adj_scipy_coo = scipy.sparse.random(127, 255, density=0.1, format='coo').astype('int32')
    test_vanilla_sddmm(adj_scipy_coo, 'x86')
    test_vanilla_sddmm(adj_scipy_coo, 'cuda')
    test_multi_head_dot_product_attention_sddmm(adj_scipy_coo, 'x86')
    # test_multi_head_dot_product_attention_sddmm(adj_scipy_coo, 'cuda')
