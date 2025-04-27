[<img height="23" src="https://raw.githubusercontent.com/lh9171338/Outline/master/icon.jpg"/>](https://github.com/lh9171338/Outline) Sparse Transformer
===
This repository is an optimized version of the [official SparseTransformer codebase](https://github.com/dvlab-research/SparseTransformer), featuring `FP16` and `BF16` support to significantly reduce GPU memory usage.

# 1. Environment

- CUDA：11.8
- PyTorch：2.0.0

# 2. Install

- install with pip
```shell
export TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6 8.9 9.0+PTX" # support different GPU architectures [optional]
export ENABLE_BF16=1 # support BF16 [optional]
python3 -m pip install SparseTransformer
```

- install from source
```shell
git clone https://github.com/lh9171338/SparseTransformer.git
cd SparseTransformer

export TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6 8.9 9.0+PTX" # support different GPU architectures [optional]
export ENABLE_BF16=1 # support BF16 [optional]
python3 setup.py install
```

# 3. Usage

Please refer to the official [README.md](https://github.com/dvlab-research/SparseTransformer/blob/master/README.md) for usage instructions.

# 4. Performance Evaluation

## 4.1 Operator Benchmarking
Tests were conducted on two attention computation modes:
- Without relative position encoding (using `attention_step1` and `attention_step2` operators)
- With relative position encoding (using `dot_prod_with_idx_all` and `attention_step2_with_rel_pos_value` operators)

Input voxel count: 268,720

(1) **Memory Consumption Comparison**

FP16 and BF16 showed identical memory usage, reducing consumption by `33%` and `35%` respectively compared to FP32.

| ID | dtype | w/o rel pos (MB) | w/ rel pos (MB) |
|:---:|:---:|:---:|:---:|
| 1 | FP32 | 580 | 622 |
| 2 | FP16 | 386 | 406 |
| 3 | BF16 | 386 | 406 |

(2) **Precision Comparison**

Maximum errors in output and gradients were measured

- Without Relative Position Encoding

| dtype | output | query grad | key grad | value grad |
|:---:|:---:|:---:|:---:|:---:|
| FP16 vs FP32 | 2.7e-02 | 6.6e-06 | 6.9e-05 | 9.9e-07 |
| BF16 vs FP32 | 3.2e-01 | 1.9e-07 | 1.9e-06 | 3.6e-08 |

- With Relative Position Encoding

| dtype | output | query grad | key grad | value grad | query table grad | key table grad | value table grad |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| FP16 vs FP32 | 3.8e-02 | 2.9e-06 | 1.1e-05 | 5.2e-07 | 1.3e-02 | 2.6e-02 | 4.0e-02 |
| BF16 vs FP32 | 2.5e-01 | 1.0e-07 | 2.0e-07 | 2.4e-08 | 1.5e-02 | 3.0e-02 | 4.1e-02 |

## 4.2 Semantic Segmentation Task Evaluation
Tests conducted using [SphereFormer]((https://github.com/dvlab-research/SphereFormer)) model

(1) **Training Mode**

With configuration parameters held constant except for the SpTr module's `dtype`, FP16/BF16 implementations demonstrate equivalent mIoU accuracy while reducing memory consumption by `31% (20.3GB => 14.1GB)`

| ID | dtype | mIoU | Memory (GB) | Time/epoch (min) |
|:---:|:---:|:---:|:---:|:---:|
| 1 | FP32 | 66.34 | 20.3 | 25 |
| 2 | FP16 | 66.18 | 14.1 | 24 |
| 3 | BF16 | 66.50 | 14.1 | 24 |

(2) **Inference Mode**

Same model evaluated with different `dtype`, both FP16 and BF16 significantly reduce memory usage by `31% (18.1GB => 15.4GB)`

| ID | dtype | mIoU | Memory (GB) | Time (s) |
|:---:|:---:|:---:|:---:|:---:|
| 1 | FP32 | 85.53 | 18.1 | 61 |
| 2 | FP16 | 85.53 | 15.4 | 58 |
| 3 | BF16 | 85.51 | 15.4 | 61 |
