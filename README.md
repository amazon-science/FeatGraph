FeatGraph: Sparse kernels for GNNs based on TVM
===============================================

Graph neural networks (GNNs) are gaining popularity in recent years as a promising approach to machine learning on graphs. Unlike traditional graph workloads where each vertex/edge is associated with a scalar, GNNs attach a feature tensor to each vertex/edge. This additional feature dimension, along with consequently more complex vertex- and edge-wise computations, has enormous implications on locality and parallelism, which existing graph processing systems fail to exploit.

To tackle the challenge, FeatGraph maps the building blocks of GNNs to generalized SpMM (sparse-dense matrix multiplication) and SDDMM (sampled dense-dense matrix multiplication) kernels, and provides high-performance implementations of these sparse kernels based on [TVM](https://tvm.apache.org/).

For more information, refer to our [SC'20 paper](https://www.csl.cornell.edu/~zhiruz/pdfs/featgraph-sc2020.pdf).
```
@article{hu2020featgraph,
  title={FeatGraph: A Flexible and Efficient Backend for Graph Neural Network Systems},
  author={Hu, Yuwei and Ye, Zihao and Wang, Minjie and Yu, Jiali and Zheng, Da and Li, Mu and Zhang, Zheng and Zhang, Zhiru and Wang, Yida},
  journal={International Conference for High Performance Computing, Networking, Storage and Analysis},
  year={2020}
}
```

## Run the code

1. Install TVM ([instructions](https://tvm.apache.org/docs/install/index.html)) and DGL ([instructions](https://docs.dgl.ai/install/index.html)).

TVM v0.7 is required. When you clone TVM:
```
git clone -b v0.7 --recursive https://github.com/apache/incubator-tvm tvm
```

2. Install FeatGraph.

```
git clone git@github.com:amazon-research/FeatGraph.git
```
```
export PYTHONPATH=/path/to/FeatGraph/python:${PYTHONPATH}
```

3. Prepare datasets.

The input to SpMM is an adjacency matrix in csr format stored as a scipy npz file; the input to SDDMM is an adjacency matrix in coo format stored as a scipy npz file.
You can run download_reddit_dataset.py under the benchmark folder to get the reddit dataset.

4. Run benchmark scripts.
```
cd benchmark
python bench_vanilla_spmm.py --dataset data/reddit_csr_float32.npz --feat-len 64 --target x86
python bench_vanilla_spmm.py --dataset data/reddit_csr_float32.npz --feat-len 64 --target cuda
python bench_vanilla_sddmm.py --dataset data/reddit_coo_float32.npz --feat-len 64 --target x86
python bench_vanilla_sddmm.py --dataset data/reddit_coo_float32.npz --feat-len 64 --target cuda
```
