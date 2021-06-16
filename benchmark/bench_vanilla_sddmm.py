import scipy
import scipy.sparse
import numpy as np
import argparse
import tvm
from tvm import te
from tvm.topi.util import get_const_tuple

from featgraph.module import VanillaSDDMMx86, VanillaSDDMMcuda


def exp_range(start, end, mul):
    while start <= end:
        yield start
        start *= mul


def bench_vanilla_sddmm_x86(adj_scipy_coo, feat_len):
    num_rows = adj_scipy_coo.shape[0]
    num_cols = adj_scipy_coo.shape[1]

    def _bench_vanilla_sddmm_x86(num_row_partitions, num_col_partitions, num_feat_partitions):
        vanilla_sddmm_module = VanillaSDDMMx86(adj_scipy_coo, num_row_partitions, num_col_partitions)
        SrcFeat = te.placeholder((num_cols, feat_len))
        DstFeat = te.placeholder((num_rows, feat_len))
        input_placeholders = [SrcFeat, DstFeat]
        compute_args = {'num_feat_partitions': num_feat_partitions}
        schedule_args = {'num_feat_partitions': num_feat_partitions}
        vanilla_sddmm_module.build(input_placeholders, compute_args, schedule_args)
        src_feat_np = np.random.random(get_const_tuple(SrcFeat.shape)).astype('float32')
        dst_feat_np = np.random.random(get_const_tuple(DstFeat.shape)).astype('float32')
        src_feat_tvm = tvm.nd.array(src_feat_np, vanilla_sddmm_module.ctx)
        dst_feat_tvm = tvm.nd.array(dst_feat_np, vanilla_sddmm_module.ctx)
        input_tvm_ndarrays = [src_feat_tvm, dst_feat_tvm]
        num_runs = 5
        tcost = vanilla_sddmm_module.measure_average_time(input_tvm_ndarrays, num_runs)
        print("average time of {} runs: {} sec".format(num_runs, tcost))

    for num_row_partitions in exp_range(1, 32, 2):
        for num_col_partitions in exp_range(1, 32, 2):
            for num_feat_partitions in exp_range(1, feat_len // 16, 2):
                print()
                print("num_row_partitions:", num_row_partitions)
                print("num_col_partitions:", num_col_partitions)
                print("num_feat_partitions:", num_feat_partitions)
                _bench_vanilla_sddmm_x86(num_row_partitions, num_col_partitions, num_feat_partitions)


def bench_vanilla_sddmm_cuda(adj_scipy_coo, feat_len):
    num_rows = adj_scipy_coo.shape[0]
    num_cols = adj_scipy_coo.shape[1]
    num_edges = adj_scipy_coo.nnz

    def _bench_vanilla_sddmm_cuda(num_cuda_blocks):
        vanilla_sddmm_module = VanillaSDDMMcuda(adj_scipy_coo)
        SrcFeat = te.placeholder((num_cols, feat_len))
        DstFeat = te.placeholder((num_rows, feat_len))
        input_placeholders = [SrcFeat, DstFeat]
        compute_args = {}
        schedule_args = {'num_cuda_blocks': num_cuda_blocks}
        vanilla_sddmm_module.build(input_placeholders, compute_args, schedule_args)
        src_feat_np = np.random.random(get_const_tuple(SrcFeat.shape)).astype('float32')
        dst_feat_np = np.random.random(get_const_tuple(DstFeat.shape)).astype('float32')
        src_feat_tvm = tvm.nd.array(src_feat_np, vanilla_sddmm_module.ctx)
        dst_feat_tvm = tvm.nd.array(dst_feat_np, vanilla_sddmm_module.ctx)
        input_tvm_ndarrays = [src_feat_tvm, dst_feat_tvm]
        num_runs = 5
        tcost = vanilla_sddmm_module.measure_average_time(input_tvm_ndarrays, num_runs)
        print("average time of {} runs: {} ms".format(num_runs, tcost * 1000))

    for num_cuda_blocks in exp_range(64, min(262144, num_edges // 32), 4):
        print()
        print("num_cuda_blocks:", num_cuda_blocks)
        _bench_vanilla_sddmm_cuda(num_cuda_blocks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="The adjacency matrix in coo format stored as a scipy npz file")
    parser.add_argument("--feat-len", type=int, default=128, help="The feature length")
    parser.add_argument("--target", type=str, default='x86', choices=['x86', 'cuda'])

    args = parser.parse_args()
    adj_scipy_coo = scipy.sparse.load_npz(args.dataset)
    assert adj_scipy_coo.format == 'coo'

    if args.target == 'x86':
        bench_vanilla_sddmm_x86(adj_scipy_coo, args.feat_len)
    elif args.target == 'cuda':
        bench_vanilla_sddmm_cuda(adj_scipy_coo, args.feat_len)
