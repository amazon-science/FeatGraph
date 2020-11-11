import scipy
import scipy.sparse
import numpy as np
import time
import tvm
from tvm import te
from tvm.topi.utils import get_const_tuple

from featgraph.module import VanillaSDDMMx86, VanillaSDDMMcuda


def test_vanilla_sddmm_x86(adj_scipy_coo):
    num_rows = adj_scipy_coo.shape[0]
    num_cols = adj_scipy_coo.shape[1]

    def _test_vanilla_sddmm_x86(feat_len, num_feat_partitions, num_row_partitions, num_col_partitions):
        vanilla_sddmm_module = VanillaSDDMMx86(adj_scipy_coo, num_row_partitions, num_col_partitions)
        SrcFeat = te.placeholder((num_cols, feat_len))
        DstFeat = te.placeholder((num_rows, feat_len))
        input_placeholders = [SrcFeat, DstFeat]
        compute_args = {'num_feat_partitions': num_feat_partitions}
        schedule_args = {'num_feat_partitions': num_feat_partitions}
        vanilla_sddmm_module.build(input_placeholders, compute_args, schedule_args)
        # run
        src_feat_np = np.random.random(get_const_tuple(SrcFeat.shape)).astype('float32')
        dst_feat_np = np.random.random(get_const_tuple(DstFeat.shape)).astype('float32')
        src_feat_tvm = tvm.nd.array(src_feat_np, vanilla_sddmm_module.ctx)
        dst_feat_tvm = tvm.nd.array(dst_feat_np, vanilla_sddmm_module.ctx)
        input_tvm_ndarrays = [src_feat_tvm, dst_feat_tvm]
        out_tvm = vanilla_sddmm_module.run(input_tvm_ndarrays).asnumpy()
        if num_row_partitions != 1 or num_col_partitions != 1:
            out_tvm = out_tvm[vanilla_sddmm_module.edge_mapping]
        # check correctness against scipy
        lhs = src_feat_np[adj_scipy_coo.col]
        rhs = dst_feat_np[adj_scipy_coo.row]
        out_scipy = (lhs * rhs).sum(axis=-1)
        np.testing.assert_allclose(out_scipy, out_tvm, rtol=1e-4, atol=1e-4)

    _test_vanilla_sddmm_x86(128, 1, 1, 1)
    _test_vanilla_sddmm_x86(128, 4, 1, 1)
    _test_vanilla_sddmm_x86(128, 4, 4, 4)


def test_vanilla_sddmm_cuda(adj_scipy_coo):
    num_rows = adj_scipy_coo.shape[0]
    num_cols = adj_scipy_coo.shape[1]

    def _test_vanilla_sddmm_cuda(feat_len, num_cuda_blocks):
        vanilla_sddmm_module = VanillaSDDMMcuda(adj_scipy_coo)
        SrcFeat = te.placeholder((num_cols, feat_len))
        DstFeat = te.placeholder((num_rows, feat_len))
        input_placeholders = [SrcFeat, DstFeat]
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
        # check correctness against scipy
        lhs = src_feat_np[adj_scipy_coo.col]
        rhs = dst_feat_np[adj_scipy_coo.row]
        out_scipy = (lhs * rhs).sum(axis=-1)
        np.testing.assert_allclose(out_scipy, out_tvm, rtol=1e-4, atol=1e-4)

    _test_vanilla_sddmm_cuda(128, 32)
    _test_vanilla_sddmm_cuda(128, 256)


if __name__ == '__main__':
    adj_scipy_coo = scipy.sparse.random(127, 255, density=0.1, format='coo').astype('int32')
    test_vanilla_sddmm_x86(adj_scipy_coo)
    test_vanilla_sddmm_cuda(adj_scipy_coo)
