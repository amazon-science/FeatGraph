import numpy as np
import tvm
from tvm import te
from tvm.topi.utils import get_const_tuple

from ..util import util_convert_csr_to_dds
from ..op import vanilla_spmm_csr_x86, schedule_vanilla_spmm_csr_x86, \
    vanilla_spmm_dds_x86, schedule_vanilla_spmm_dds_x86, \
    vanilla_spmm_csr_cuda, schedule_vanilla_spmm_csr_cuda


class SpMMbase():
    """The base class for SpMM-like computation kernels."""
    def __init__(self, adj_scipy, num_col_partitions=1):
        """Doing 1D graph partitioning (src vertex partitioning) during init.

        Parameters
        ----------
        adj_scipy : scipy.sparse.coo_matrix or scipy.sparse.csr_matrix
            The input scipy sparse matrix

        num_col_partitions : int
            Number of partitions along the col dimension (src vertices)
        """
        # Use csr format in SpMM-like kernels
        if adj_scipy.format != 'csr':
            adj_scipy_csr = adj_scipy.tocsr()
        else:
            adj_scipy_csr = adj_scipy
        self._num_rows = adj_scipy_csr.shape[0]
        self._num_cols = adj_scipy_csr.shape[1]
        assert num_col_partitions >= 1, "num_col_partitions should be larger than or equal to 1"
        self._num_col_partitions = num_col_partitions
        # To be updated in self._register
        self._target = None
        self._ctx = None
        self._compute_func = None
        self._schedule_func = None
        self._register()
        # 1D graph partitioning
        if self._num_col_partitions > 1:
            adj_s1_pos, adj_s1_idx, adj_vals = self._preprocess_adj(adj_scipy_csr, self._num_col_partitions)
            self._adj_s1_pos = adj_s1_pos
            self._adj_s1_idx = adj_s1_idx
            self._adj_vals = adj_vals
            self._adj_s1_pos_placeholder = te.placeholder(shape=self._adj_s1_pos.shape, \
                dtype=str(self._adj_s1_pos.dtype), name='adj_s1_pos_placeholder')
            self._adj_s1_idx_placeholder = te.placeholder(shape=self._adj_s1_idx.shape, \
                dtype=str(self._adj_s1_idx.dtype), name='adj_s1_idx_placeholder')
            self._adj_vals_placeholder = te.placeholder(shape=self._adj_vals.shape, \
                dtype=str(self._adj_vals.dtype), name='adj_vals_placeholder')
            self._adj_s1_pos_tvm = tvm.nd.array(self._adj_s1_pos, ctx=self._ctx)
            self._adj_s1_idx_tvm = tvm.nd.array(self._adj_s1_idx, ctx=self._ctx)
            self._adj_vals_tvm = tvm.nd.array(self._adj_vals, ctx=self._ctx)
            self._adj_d1_size = self._num_col_partitions
            self._adj_d2_size = self._num_rows + 1
        else:
            self._adj_indptr = adj_scipy_csr.indptr
            self._adj_indices = adj_scipy_csr.indices
            self._adj_vals = adj_scipy_csr.data
            self._adj_indptr_placeholder = te.placeholder(shape=self._adj_indptr.shape, \
                dtype=str(self._adj_indptr.dtype), name='adj_indptr_placeholder')
            self._adj_indices_placeholder = te.placeholder(shape=self._adj_indices.shape, \
                dtype=str(self._adj_indices.dtype), name='adj_indices_placeholder')
            self._adj_vals_placeholder = te.placeholder(shape=self._adj_vals.shape, \
                dtype=str(self._adj_vals.dtype), name='adj_vals_placeholder')
            self._adj_indptr_tvm = tvm.nd.array(self._adj_indptr, ctx=self._ctx)
            self._adj_indices_tvm = tvm.nd.array(self._adj_indices, ctx=self._ctx)
            self._adj_vals_tvm = tvm.nd.array(self._adj_vals, ctx=self._ctx)
        # To be updated in self.build
        self._func = None
        # To be updated in self.run
        self.out_tvm = None

    def _preprocess_adj(self, adj_scipy_csr, num_col_partitions=1):
        return util_convert_csr_to_dds(adj_scipy_csr, num_col_partitions)

    def _register(self):
        raise NotImplementedError("Please register target, ctx, compute_func, and schedule_func.")

    def build(self, input_placeholders, compute_args, schedule_args):
        """Build tvm func, update self._func inplace.

        Parameters
        ----------
        input_placeholders : list of te.placeholder
            The required input tvm placeholders other than adj (which has been passed in during self.init)

        compute_args : dict
            Arguments required for compute_func, e.g., num_feat_partitions

        schedule_args : dict
            Arguments required for schedule_func, e.g., num_cuda_blocks
        """
        if self._num_col_partitions > 1:
            out_placeholder = self._compute_func(*input_placeholders, self._adj_s1_pos_placeholder, \
                self._adj_s1_idx_placeholder, self._adj_vals_placeholder,
                self._adj_d1_size, self._adj_d2_size, **compute_args)  # use ** to unpack dict into kwargs
            s = self._schedule_func(out_placeholder, **schedule_args)
            self._func = tvm.build(s, [*input_placeholders, self._adj_s1_pos_placeholder, \
                self._adj_s1_idx_placeholder, self._adj_vals_placeholder, out_placeholder], target=self._target)
            self.out_tvm = tvm.nd.array(np.zeros(shape=get_const_tuple(out_placeholder.shape), \
                dtype=str(out_placeholder.dtype)), ctx=self._ctx)
        else:
            out_placeholder = self._compute_func(*input_placeholders, self._adj_indptr_placeholder, \
                self._adj_indices_placeholder, self._adj_vals_placeholder, **compute_args)
            s = self._schedule_func(out_placeholder, **schedule_args)
            self._func = tvm.build(s, [*input_placeholders, self._adj_indptr_placeholder, \
                self._adj_indices_placeholder, self._adj_vals_placeholder, out_placeholder], target=self._target)
            self.out_tvm = tvm.nd.array(np.zeros(shape=get_const_tuple(out_placeholder.shape), \
                dtype=str(out_placeholder.dtype)), ctx=self._ctx)

    def lower_to_ir(self, input_placeholders, compute_args, schedule_args):
        """Return the IR. This can be useful for debug.

        Parameters
        ----------
        input_placeholders : list of te.placeholder
            The required input tvm placeholders other than adj (which has been passed in during self.init)

        compute_args : dict
            Arguments required for compute_func, e.g., num_feat_partitions

        schedule_args : dict
            Arguments required for schedule_func, e.g., num_cuda_blocks
        """
        if self._num_col_partitions > 1:
            out_placeholder = self._compute_func(*input_placeholders, self._adj_s1_pos_placeholder, \
                self._adj_s1_idx_placeholder, self._adj_vals_placeholder,
                self._adj_d1_size, self._adj_d2_size, **compute_args)
            s = self._schedule_func(out_placeholder, **schedule_args)
            ir = tvm.lower(s, [*input_placeholders, self._adj_s1_pos_placeholder, \
                self._adj_s1_idx_placeholder, self._adj_vals_placeholder, out_placeholder], simple_mode=True)
        else:
            out_placeholder = self._compute_func(*input_placeholders, self._adj_indptr_placeholder, \
                self._adj_indices_placeholder, self._adj_vals_placeholder, **compute_args)
            s = self._schedule_func(out_placeholder, **schedule_args)
            ir = tvm.lower(s, [*input_placeholders, self._adj_indptr_placeholder, \
                self._adj_indices_placeholder, self._adj_vals_placeholder, out_placeholder], simple_mode=True)
        return ir

    def run(self, input_tvm_ndarrays):
        """Run the built func with the given inputs input_tvm_ndarrays.

        Parameters
        ----------
        input_tvm_ndarrays : list of tvm.ndarray
            The required input tvm ndarrays other than adj (which has been created during self.build)

        Returns
        -------
        self.out_tvm: tvm.ndarray
            The output tvm ndarray
        """
        if self._num_col_partitions > 1:
            self._func(*input_tvm_ndarrays, self._adj_s1_pos_tvm, self._adj_s1_idx_tvm, \
                self._adj_vals_tvm, self.out_tvm)
        else:
            self._func(*input_tvm_ndarrays, self._adj_indptr_tvm, self._adj_indices_tvm, \
                self._adj_vals_tvm, self.out_tvm)
        return self.out_tvm

    def measure_average_time(self, input_tvm_ndarrays, num_runs):
        """Measure the run time of the built module using tvm time_evaluator.

        Parameters
        ----------
        input_tvm_ndarrays : list of tvm.ndarray
            The required input tvm ndarrays other than adj (which has been created during self.build)

        int : num_runs
            The number of runs

        Returns
        -------
        tcost: float32
            The average run time
        """
        timer = self._func.time_evaluator(self._func.entry_name, ctx=self._ctx, number=num_runs)
        if self._num_col_partitions > 1:
            tcost = timer(*input_tvm_ndarrays, self._adj_s1_pos_tvm, \
                self._adj_s1_idx_tvm, self._adj_vals_tvm, self.out_tvm).mean
        else:
            tcost = timer(*input_tvm_ndarrays, self._adj_indptr_tvm, \
                self._adj_indices_tvm, self._adj_vals_tvm, self.out_tvm).mean
        return tcost

    @property
    def ctx(self):
        return self._ctx

    @property
    def target(self):
        return self._target


class VanillaSpMMx86(SpMMbase):
    def __init__(self, adj_scipy, num_col_partitions=1):
        super(VanillaSpMMx86, self).__init__(adj_scipy, num_col_partitions)

    def _register(self):
        self._target = 'llvm'
        self._ctx = tvm.cpu(0)
        if self._num_col_partitions > 1:
            self._compute_func = vanilla_spmm_dds_x86
            self._schedule_func = schedule_vanilla_spmm_dds_x86
        else:
            self._compute_func = vanilla_spmm_csr_x86
            self._schedule_func = schedule_vanilla_spmm_csr_x86


class VanillaSpMMcuda(SpMMbase):
    def __init__(self, adj_scipy):
        super(VanillaSpMMcuda, self).__init__(adj_scipy, num_col_partitions=1)

    def _register(self):
        self._target = 'cuda'
        self._ctx = tvm.gpu(0)
        self._compute_func = vanilla_spmm_csr_cuda
        self._schedule_func = schedule_vanilla_spmm_csr_cuda
