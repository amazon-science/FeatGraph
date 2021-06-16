import scipy
import scipy.sparse
import numpy as np
import argparse
import tvm
from tvm import te
from tvm.topi.util import get_const_tuple

from featgraph.module import VanillaSpMMx86, VanillaSpMMcuda


def exp_range(start, end, mul):
    while start <= end:
        yield start
        start *= mul


def bench_vanilla_spmm_x86(adj_scipy_csr, feat_len):
    num_rows = adj_scipy_csr.shape[0]
    num_cols = adj_scipy_csr.shape[1]

    def _bench_vanilla_spmm_x86(num_col_partitions, num_feat_partitions):
        vanilla_spmm_module = VanillaSpMMx86(adj_scipy_csr, num_col_partitions)
        SrcFeat = te.placeholder((num_cols, feat_len))
        input_placeholders = [SrcFeat]
        compute_args = {'num_feat_partitions': num_feat_partitions}
        schedule_args = {}
        vanilla_spmm_module.build(input_placeholders, compute_args, schedule_args)
        src_feat_np = np.random.random(get_const_tuple(SrcFeat.shape)).astype('float32')
        src_feat_tvm = tvm.nd.array(src_feat_np, vanilla_spmm_module.ctx)
        input_tvm_ndarrays = [src_feat_tvm]
        num_runs = 5
        tcost = vanilla_spmm_module.measure_average_time(input_tvm_ndarrays, num_runs)
        print("average time of {} runs: {} sec".format(num_runs, tcost))

    for num_col_partitions in exp_range(1, 32, 2):
        for num_feat_partitions in exp_range(1, feat_len // 16, 2):
            print()
            print("num_col_partitions:", num_col_partitions)
            print("num_feat_partitions:", num_feat_partitions)
            _bench_vanilla_spmm_x86(num_col_partitions, num_feat_partitions)


def bench_vanilla_spmm_cuda(adj_scipy_csr, feat_len):
    num_rows = adj_scipy_csr.shape[0]
    num_cols = adj_scipy_csr.shape[1]

    def _bench_vanilla_spmm_cuda(num_cuda_blocks, num_threads_per_cuda_block):
        vanilla_spmm_module = VanillaSpMMcuda(adj_scipy_csr)
        SrcFeat = te.placeholder((num_cols, feat_len))
        input_placeholders = [SrcFeat]
        compute_args = {}
        schedule_args = {'num_cuda_blocks': num_cuda_blocks,
                         'num_threads_per_cuda_block': num_threads_per_cuda_block}
        vanilla_spmm_module.build(input_placeholders, compute_args, schedule_args)
        src_feat_np = np.random.random(get_const_tuple(SrcFeat.shape)).astype('float32')
        src_feat_tvm = tvm.nd.array(src_feat_np, vanilla_spmm_module.ctx)
        input_tvm_ndarrays = [src_feat_tvm]
        num_runs = 5
        tcost = vanilla_spmm_module.measure_average_time(input_tvm_ndarrays, num_runs)
        print("average time of {} runs: {} ms".format(num_runs, tcost * 1000))

    for num_cuda_blocks in exp_range(64, num_rows, 4):
        for num_threads_per_cuda_block in exp_range(min(feat_len, 32), feat_len, 2):
            print()
            print("num_cuda_blocks:", num_cuda_blocks)
            print("num_threads_per_cuda_block:", num_threads_per_cuda_block)
            _bench_vanilla_spmm_cuda(num_cuda_blocks, num_threads_per_cuda_block)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="The adjacency matrix in csr format stored as a scipy npz file")
    parser.add_argument("--feat-len", type=int, default=128, help="The feature length")
    parser.add_argument("--target", type=str, default='x86', choices=['x86', 'cuda'])

    args = parser.parse_args()
    adj_scipy_csr = scipy.sparse.load_npz(args.dataset)
    assert adj_scipy_csr.format == 'csr'

    if args.target == 'x86':
        bench_vanilla_spmm_x86(adj_scipy_csr, args.feat_len)
    elif args.target == 'cuda':
        bench_vanilla_spmm_cuda(adj_scipy_csr, args.feat_len)
