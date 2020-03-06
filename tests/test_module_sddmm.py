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
        adj_row_num_partition = 4
        adj_col_num_partition = 4
        module = VanillaSDDMMx86
    elif target == 'cuda':
        adj_row_num_partition = 1
        adj_col_num_partition = 1
        module = VanillaSDDMMcuda
    else:
        raise RuntimeError("invalid target")
    vanilla_sddmm_module = module(adj_scipy_coo, adj_row_num_partition, adj_col_num_partition)

    # tvm func is built for a specific feat_len and num_feat_partitions
    feat_len = 128
    SrcFeat = tvm.placeholder((num_rows, feat_len))
    DstFeat = tvm.placeholder((num_cols, feat_len))
    input_placeholders = [SrcFeat, DstFeat]
    if target == 'x86':
        num_feat_partitions = 4
        compute_args = {'num_feat_partitions': num_feat_partitions}
        schedule_args = {'num_feat_partitions': num_feat_partitions}
    elif target == 'cuda':
        num_cuda_blocks = 4096
        num_threads_per_cuda_block = 64
        compute_args = {}
        schedule_args = {'num_cuda_blocks': num_cuda_blocks,
                         'num_threads_per_cuda_block': num_threads_per_cuda_block}
    else:
        raise RuntimeError("invalid target")
    vanilla_sddmm_module.build(input_placeholders, compute_args, schedule_args)

    # run
    src_feat_np = np.random.random(get_const_tuple(SrcFeat.shape)).astype('float32')
    dst_feat_np = np.random.random(get_const_tuple(DstFeat.shape)).astype('float32')
    src_feat_tvm = tvm.nd.array(src_feat_np, vanilla_sddmm_module.ctx)
    dst_feat_tvm = tvm.nd.array(dst_feat_np, vanilla_sddmm_module.ctx)
    feat_tvm_ndarrays = [src_feat_tvm, dst_feat_tvm]
    out_tvm = vanilla_sddmm_module.run(feat_tvm_ndarrays).asnumpy()
    # be careful here
    if target == 'x86':
        out_tvm = out_tvm[vanilla_sddmm_module.edge_mapping]

    # check correctness against scipy
    lhs = src_feat_np[adj_scipy_coo.row]
    rhs = dst_feat_np[adj_scipy_coo.col]
    out_scipy = (lhs * rhs).sum(axis=-1)
    np.testing.assert_allclose(out_scipy, out_tvm, rtol=1e-4, atol=1e-4)


def test_multi_head_dot_product_attention_sddmm(adj_scipy_coo, target):
    num_rows = adj_scipy_coo.shape[0]
    num_cols = adj_scipy_coo.shape[1]

    # doing 2D graph partitioning during initialization
    # note that 2D partitioning is mainly useful for CPU since it optimizes cache
    if target == 'x86':
        adj_row_num_partition = 4
        adj_col_num_partition = 4
        module = MultiHeadSDDMMx86
    elif target == 'cuda':
        adj_row_num_partition = 1
        adj_col_num_partition = 1
        module = MultiHeadSDDMMcuda
    else:
        raise RuntimeError("invalid target")
    multi_head_sddmm_module = module(adj_scipy_coo, adj_row_num_partition, adj_col_num_partition)

    # tvm func is built for a specific num_heads, num_head_partitions, feat_len, num_feat_partitions
    num_heads = 16
    feat_len = 64
    SrcFeat = tvm.placeholder((num_rows, num_heads, feat_len))
    DstFeat = tvm.placeholder((num_cols, num_heads, feat_len))
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
    else:
        raise RuntimeError("invalid target")
    multi_head_sddmm_module.build(input_placeholders, compute_args, schedule_args)

    # run
    src_feat_np = np.random.random(get_const_tuple(SrcFeat.shape)).astype('float32')
    dst_feat_np = np.random.random(get_const_tuple(DstFeat.shape)).astype('float32')
    src_feat_tvm = tvm.nd.array(src_feat_np, multi_head_sddmm_module.ctx)
    dst_feat_tvm = tvm.nd.array(dst_feat_np, multi_head_sddmm_module.ctx)
    feat_tvm_ndarrays = [src_feat_tvm, dst_feat_tvm]
    out_tvm = multi_head_sddmm_module.run(feat_tvm_ndarrays).asnumpy()
    # be careful here
    if target == 'x86':
        out_tvm = out_tvm[multi_head_sddmm_module.edge_mapping]

    # check correctness against scipy
    lhs = src_feat_np[adj_scipy_coo.row]
    rhs = dst_feat_np[adj_scipy_coo.col]
    out_scipy = (lhs * rhs).sum(axis=-1)
    np.testing.assert_allclose(out_scipy, out_tvm, rtol=1e-4, atol=1e-4)


if __name__ == '__main__':
    adj_scipy_coo = scipy.sparse.random(127, 255, density=0.1, format='coo').astype('int32')
    test_vanilla_sddmm(adj_scipy_coo, 'x86')
    test_multi_head_dot_product_attention_sddmm(adj_scipy_coo, 'x86')
