import scipy
import scipy.sparse
import numpy as np
import time
import tvm
from tvm import te
from tvm.topi.util import get_const_tuple

from featgraph.module import VanillaSpMMx86, VanillaSpMMcuda


def test_vanilla_spmm_x86(adj_scipy_csr):
    num_rows = adj_scipy_csr.shape[0]
    num_cols = adj_scipy_csr.shape[1]

    def _test_vanilla_spmm_x86(feat_len, num_col_partitions, num_feat_partitions):
        vanilla_spmm_module = VanillaSpMMx86(adj_scipy_csr, num_col_partitions)
        SrcFeat = te.placeholder((num_cols, feat_len))
        input_placeholders = [SrcFeat]
        compute_args = {'num_feat_partitions': num_feat_partitions}
        schedule_args = {}
        vanilla_spmm_module.build(input_placeholders, compute_args, schedule_args)
        # run
        src_feat_np = np.random.random(get_const_tuple(SrcFeat.shape)).astype('float32')
        src_feat_tvm = tvm.nd.array(src_feat_np, vanilla_spmm_module.ctx)
        input_tvm_ndarrays = [src_feat_tvm]
        out_tvm = vanilla_spmm_module.run(input_tvm_ndarrays).asnumpy()
        # check correctness against scipy
        out_scipy = adj_scipy_csr.dot(src_feat_np)
        np.testing.assert_allclose(out_scipy, out_tvm, rtol=1e-4, atol=1e-4)

    _test_vanilla_spmm_x86(128, 1, 1)
    _test_vanilla_spmm_x86(128, 4, 1)
    _test_vanilla_spmm_x86(128, 4, 4)


def test_vanilla_spmm_cuda(adj_scipy_csr):
    num_rows = adj_scipy_csr.shape[0]
    num_cols = adj_scipy_csr.shape[1]

    def _test_vanilla_spmm_cuda(feat_len, num_threads_per_cuda_block):
        vanilla_spmm_module = VanillaSpMMcuda(adj_scipy_csr)
        SrcFeat = te.placeholder((num_cols, feat_len))
        input_placeholders = [SrcFeat]
        compute_args = {}
        schedule_args = {'num_threads_per_cuda_block': num_threads_per_cuda_block}
        vanilla_spmm_module.build(input_placeholders, compute_args, schedule_args)
        # run
        src_feat_np = np.random.random(get_const_tuple(SrcFeat.shape)).astype('float32')
        src_feat_tvm = tvm.nd.array(src_feat_np, vanilla_spmm_module.ctx)
        input_tvm_ndarrays = [src_feat_tvm]
        out_tvm = vanilla_spmm_module.run(input_tvm_ndarrays).asnumpy()
        # check correctness against scipy
        out_scipy = adj_scipy_csr.dot(src_feat_np)
        np.testing.assert_allclose(out_scipy, out_tvm, rtol=1e-4, atol=1e-4)

    _test_vanilla_spmm_cuda(128, 32)
    _test_vanilla_spmm_cuda(128, 128)
    _test_vanilla_spmm_cuda(128, None)


if __name__ == '__main__':
    adj_scipy_csr = scipy.sparse.random(127, 255, density=0.1, format='csr').astype('float32')
    test_vanilla_spmm_x86(adj_scipy_csr)
    test_vanilla_spmm_cuda(adj_scipy_csr)
