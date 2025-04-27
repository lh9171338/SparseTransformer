#ifndef _ATTENTION_CUDA_KERNEL
#define _ATTENTION_CUDA_KERNEL
#include <torch/extension.h>
#include <cuda_fp16.h> // 包含 __half 的定义
#ifdef ENABLE_BF16
#include <cuda_bf16.h> // 包含 __nv_bfloat16 的定义
#endif

void attention_step1_forward_cuda(int N_q, int N_k, int M, int h, int hdim,
    const unsigned int n_max, at::Tensor q_tensor, at::Tensor k_tensor,
    at::Tensor index0_tensor, at::Tensor index1_tensor, at::Tensor attn_tensor);
void attention_step1_backward_cuda(int N, int M, int h, int hdim, const unsigned int n_max,
    at::Tensor grad_out_tensor, at::Tensor index0_tensor, at::Tensor index0_tensor_offsets,
    at::Tensor index1_tensor, at::Tensor index1_tensor_offsets, at::Tensor q_tensor, at::Tensor k_tensor,
    at::Tensor grad_q_tensor, at::Tensor grad_k_tensor);

void attention_step2_forward_cuda(int N, int M, int h, int hdim, int n_max, at::Tensor attn_tensor,
    at::Tensor v_tensor, at::Tensor index0_offsets_tensor, at::Tensor index1_tensor, at::Tensor output_tensor);
void attention_step2_backward_cuda(int N, int M, int h, int hdim, int n_max, at::Tensor grad_out_tensor,
    at::Tensor index0_tensor, at::Tensor index0_offsets_tensor, at::Tensor index1_tensor,
    at::Tensor index1_offsets_tensor, at::Tensor attn_tensor, at::Tensor v_tensor,
    at::Tensor grad_attn_tensor, at::Tensor grad_v_tensor);

#ifdef __cplusplus
extern "C" {
#endif

void attention_step1_forward_cuda_launcher_fp32(int N_q, int N_k, int M, int h, int hdim,
    const unsigned int n_max, const float *q, const float *k, const int *index0, const int *index1, float *attn);
void attention_step1_backward_cuda_launcher_fp32(int N, int M, int h, int hdim, const unsigned int n_max,
    const float *grad_out, const int *index0, const int *index0_offsets, const int *index1,
    const int *index1_offsets, const float *q, const float *k, float *grad_q, float *grad_k);

void attention_step2_forward_cuda_launcher_fp32(int N, int M, const int h, int hdim, int n_max,
    const float *attn, const float *v, const int *index0_offsets, const int *index1, float *output);
void attention_step2_backward_cuda_launcher_fp32(int N, int M, int h, int hdim, int n_max,
    const float *grad_out, const int *index0, const int *index0_offsets, const int *index1,
    const int *index1_offsets, const float *attn, const float *v, float *grad_attn, float *grad_v);

void attention_step1_forward_cuda_launcher_fp16(int N_q, int N_k, int M, int h, int hdim,
    const unsigned int n_max, const __half *q, const __half *k, const int *index0, const int *index1, __half *attn);
void attention_step1_backward_cuda_launcher_fp16(int N, int M, int h, int hdim, const unsigned int n_max,
    const __half *grad_out, const int *index0, const int *index0_offsets, const int *index1,
    const int *index1_offsets, const __half *q, const __half *k, __half *grad_q, __half *grad_k);

void attention_step2_forward_cuda_launcher_fp16(int N, int M, const int h, int hdim, int n_max,
    const __half *attn, const __half *v, const int *index0_offsets, const int *index1, __half *output);
void attention_step2_backward_cuda_launcher_fp16(int N, int M, int h, int hdim, int n_max,
    const __half *grad_out, const int *index0, const int *index0_offsets, const int *index1,
    const int *index1_offsets, const __half *attn, const __half *v, __half *grad_attn, __half *grad_v);

#ifdef ENABLE_BF16
void attention_step1_forward_cuda_launcher_bf16(int N_q, int N_k, int M, int h, int hdim,
    const unsigned int n_max, const __nv_bfloat16 *q, const __nv_bfloat16 *k, const int *index0, const int *index1, __nv_bfloat16 *attn);
void attention_step1_backward_cuda_launcher_bf16(int N, int M, int h, int hdim, const unsigned int n_max,
    const __nv_bfloat16 *grad_out, const int *index0, const int *index0_offsets, const int *index1,
    const int *index1_offsets, const __nv_bfloat16 *q, const __nv_bfloat16 *k, __nv_bfloat16 *grad_q, __nv_bfloat16 *grad_k);

void attention_step2_forward_cuda_launcher_bf16(int N, int M, const int h, int hdim, int n_max,
    const __nv_bfloat16 *attn, const __nv_bfloat16 *v, const int *index0_offsets, const int *index1, __nv_bfloat16 *output);
void attention_step2_backward_cuda_launcher_bf16(int N, int M, int h, int hdim, int n_max,
    const __nv_bfloat16 *grad_out, const int *index0, const int *index0_offsets, const int *index1,
    const int *index1_offsets, const __nv_bfloat16 *attn, const __nv_bfloat16 *v, __nv_bfloat16 *grad_attn, __nv_bfloat16 *grad_v);
#endif

#ifdef __cplusplus
}
#endif
#endif
