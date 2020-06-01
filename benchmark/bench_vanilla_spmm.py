import scipy
import scipy.sparse
import numpy as np
import time
import tvm
from topi.util import get_const_tuple

from featgraph.module import VanillaSpMMx86, VanillaSpMMcuda


def bench_vanilla_spmm_x86(adj_scipy_csr):
    num_rows = adj_scipy_csr.shape[0]
    num_cols = adj_scipy_csr.shape[1]

    def _bench_vanilla_spmm_x86(num_col_partitions, feat_len, num_feat_partitions):
        vanilla_spmm_module = VanillaSpMMx86(adj_scipy_csr, num_col_partitions)
        SrcFeat = tvm.placeholder((num_cols, feat_len))
        input_placeholders = [SrcFeat]
        compute_args = {'num_feat_partitions': num_feat_partitions}
        schedule_args = {}
        vanilla_spmm_module.build(input_placeholders, compute_args, schedule_args)
        src_feat_np = np.random.random(get_const_tuple(SrcFeat.shape)).astype('float32')
        src_feat_tvm = tvm.nd.array(src_feat_np, vanilla_spmm_module.ctx)
        input_tvm_ndarrays = [src_feat_tvm]
        num_runs = 2
        tcost = vanilla_spmm_module.measure_average_time(input_tvm_ndarrays, num_runs)
        print("average time of {} runs: {} sec".format(num_runs, tcost))

    feat_len = 128
    for num_col_partitions in [1, 2, 4]:
        for num_feat_partitions in [1, 2, 4]:
            print("num_col_partitions:", num_col_partitions)
            print("num_feat_partitions:", num_feat_partitions)
            _bench_vanilla_spmm_x86(num_col_partitions, feat_len, num_feat_partitions)


def bench_vanilla_spmm_cuda(adj_scipy_csr):
    num_rows = adj_scipy_csr.shape[0]
    num_cols = adj_scipy_csr.shape[1]

    def _bench_vanilla_spmm_cuda(feat_len, num_threads_per_cuda_block):
        vanilla_spmm_module = VanillaSpMMcuda(adj_scipy_csr)
        SrcFeat = tvm.placeholder((num_cols, feat_len))
        input_placeholders = [SrcFeat]
        compute_args = {}
        schedule_args = {'num_threads_per_cuda_block': num_threads_per_cuda_block}
        vanilla_spmm_module.build(input_placeholders, compute_args, schedule_args)
        src_feat_np = np.random.random(get_const_tuple(SrcFeat.shape)).astype('float32')
        src_feat_tvm = tvm.nd.array(src_feat_np, vanilla_spmm_module.ctx)
        input_tvm_ndarrays = [src_feat_tvm]
        num_runs = 2
        tcost = vanilla_spmm_module.measure_average_time(input_tvm_ndarrays, num_runs)
        print("average time of {} runs: {} sec".format(num_runs, tcost))

    feat_len = 256
    for num_threads_per_cuda_block in [16, 32, 64, 128]:
        print("num_threads_per_cuda_block:", num_threads_per_cuda_block)
        _bench_vanilla_spmm_cuda(feat_len, num_threads_per_cuda_block)


if __name__ == '__main__':
    adj_scipy_csr = scipy.sparse.load_npz("/home/ubuntu/efs/data/sparse_matrix_graph/reddit_200K_115M_csr_float32.npz")
    bench_vanilla_spmm_x86(adj_scipy_csr)
    bench_vanilla_spmm_cuda(adj_scipy_csr)
