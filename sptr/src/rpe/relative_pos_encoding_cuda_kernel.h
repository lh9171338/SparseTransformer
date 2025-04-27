#ifndef _RPE_CUDA_KERNEL
#define _RPE_CUDA_KERNEL
#include <torch/extension.h>
#include <cuda_fp16.h> // 包含 __half 的定义
#ifdef ENABLE_BF16
#include <cuda_bf16.h> // 包含 __nv_bfloat16 的定义
#endif

void dot_prod_with_idx_forward_cuda(int N, int M, int h, int hdim, int n_max, const int L, at::Tensor q_tensor,
    at::Tensor index_q_tensor, at::Tensor index_q_offsets_tensor, at::Tensor k_tensor, at::Tensor index_k_tensor,
    at::Tensor table_q_tensor, at::Tensor table_k_tensor, at::Tensor rel_idx_tensor, at::Tensor output_tensor);
void dot_prod_with_idx_backward_cuda(int N, int M, int h, int hdim, int n_max, const int L, at::Tensor grad_out_tensor,
    at::Tensor q_tensor, at::Tensor index_q_offsets_tensor, at::Tensor k_tensor, at::Tensor index_k_offsets_tensor,
    at::Tensor index_k_tensor, at::Tensor table_q_tensor, at::Tensor table_k_tensor, at::Tensor rel_idx_tensor,
    at::Tensor grad_q_tensor, at::Tensor grad_k_tensor, at::Tensor grad_table_q_tensor,
    at::Tensor grad_table_k_tensor);

void dot_prod_with_idx_all_forward_cuda(int N, int M, int h, int hdim, int n_max, const int L, at::Tensor q_tensor,
    at::Tensor index_q_tensor, at::Tensor index_q_offsets_tensor, at::Tensor k_tensor, at::Tensor index_k_tensor,
    at::Tensor table_q_tensor, at::Tensor table_k_tensor, at::Tensor rel_idx_tensor, at::Tensor output_tensor);
void dot_prod_with_idx_all_backward_cuda(int N, int M, int h, int hdim, int n_max, const int L,
    at::Tensor grad_out_tensor,
    at::Tensor q_tensor, at::Tensor index_q_offsets_tensor, at::Tensor k_tensor, at::Tensor index_k_offsets_tensor,
    at::Tensor index_k_tensor, at::Tensor table_q_tensor, at::Tensor table_k_tensor, at::Tensor rel_idx_tensor,
    at::Tensor grad_q_tensor, at::Tensor grad_k_tensor, at::Tensor grad_table_q_tensor,
    at::Tensor grad_table_k_tensor);

void attention_step2_with_rel_pos_value_forward_cuda(int N, int M, int h, int hdim, int n_max, at::Tensor attn_tensor,
    at::Tensor v_tensor, at::Tensor index0_offsets_tensor, at::Tensor index1_tensor, at::Tensor table_tensor,
    at::Tensor rel_idx_tensor, at::Tensor output_tensor);
void attention_step2_with_rel_pos_value_backward_cuda(int N, int M, int h, int hdim, int L, int n_max,
    at::Tensor grad_out_tensor, at::Tensor index0_tensor, at::Tensor index0_offsets_tensor, at::Tensor index1_tensor,
    at::Tensor index1_offsets_tensor, at::Tensor attn_tensor, at::Tensor v_tensor, at::Tensor table_tensor,
    at::Tensor rel_idx_tensor, at::Tensor grad_attn_tensor, at::Tensor grad_v_tensor, at::Tensor grad_table_tensor);

#ifdef __cplusplus
extern "C" {
#endif

void dot_prod_with_idx_forward_cuda_launcher_fp32(int N, int M, int h, int hdim, int n_max, const int L,
    const float *q, const int *index_q, const int *index_q_offsets, const float *k, const int *index_k,
    const float *table_q, const float *table_k, const char *rel_idx, float *output);
void dot_prod_with_idx_backward_cuda_launcher_fp32(int N, int M, int h, int hdim, int n_max, const int L,
    const float *grad_out, const float *q, const int *index_q_offsets, const float *k, const int *index_k_offsets,
    const int *index_k, const float *table_q, const float *table_k, const char *rel_idx, float *grad_q,
    float *grad_k, float *grad_table_q, float *grad_table_k);

void dot_prod_with_idx_all_forward_cuda_launcher_fp32(int N, int M, int h, int hdim, int n_max, const int L,
    const float *q, const int *index_q, const int *index_q_offsets, const float *k, const int *index_k,
    const float *table_q, const float *table_k, const char *rel_idx, float *output);
void dot_prod_with_idx_all_backward_cuda_launcher_fp32(int N, int M, int h, int hdim, int n_max, const int L,
    const float *grad_out, const float *q, const int *index_q_offsets, const float *k, const int *index_k_offsets,
    const int *index_k, const float *table_q, const float *table_k, const char *rel_idx, float *grad_q,
    float *grad_k, float *grad_table_q, float *grad_table_k);

void attention_step2_with_rel_pos_value_forward_cuda_launcher_fp32(int N, int M, int h, int hdim, int n_max,
    const float *attn, const float *v, const int *index0_offsets, const int *index1, const float *table,
    const char *rel_idx, float *output);
void attention_step2_with_rel_pos_value_backward_cuda_launcher_fp32(int N, int M, int h, int hdim, int L,
    int n_max, const float *grad_out, const int *index0, const int *index0_offsets, const int *index1,
    const int *index1_offsets, const float *attn, const float *v, const float *table, const char *rel_idx,
    float *grad_attn, float *grad_v, float *grad_table);

void dot_prod_with_idx_forward_cuda_launcher_fp16(int N, int M, int h, int hdim, int n_max, const int L,
    const __half *q, const int *index_q, const int *index_q_offsets, const __half *k, const int *index_k,
    const __half *table_q, const __half *table_k, const char *rel_idx, __half *output);
void dot_prod_with_idx_backward_cuda_launcher_fp16(int N, int M, int h, int hdim, int n_max, const int L,
    const __half *grad_out, const __half *q, const int *index_q_offsets, const __half *k, const int *index_k_offsets,
    const int *index_k, const __half *table_q, const __half *table_k, const char *rel_idx, __half *grad_q,
    __half *grad_k, __half *grad_table_q, __half *grad_table_k);

void dot_prod_with_idx_all_forward_cuda_launcher_fp16(int N, int M, int h, int hdim, int n_max, const int L,
    const __half *q, const int *index_q, const int *index_q_offsets, const __half *k, const int *index_k,
    const __half *table_q, const __half *table_k, const char *rel_idx, __half *output);
void dot_prod_with_idx_all_backward_cuda_launcher_fp16(int N, int M, int h, int hdim, int n_max, const int L,
    const __half *grad_out, const __half *q, const int *index_q_offsets, const __half *k, const int *index_k_offsets,
    const int *index_k, const __half *table_q, const __half *table_k, const char *rel_idx, __half *grad_q,
    __half *grad_k, __half *grad_table_q, __half *grad_table_k);

void attention_step2_with_rel_pos_value_forward_cuda_launcher_fp16(int N, int M, int h, int hdim, int n_max,
    const __half *attn, const __half *v, const int *index0_offsets, const int *index1, const __half *table,
    const char *rel_idx, __half *output);
void attention_step2_with_rel_pos_value_backward_cuda_launcher_fp16(int N, int M, int h, int hdim, int L,
    int n_max, const __half *grad_out, const int *index0, const int *index0_offsets, const int *index1,
    const int *index1_offsets, const __half *attn, const __half *v, const __half *table, const char *rel_idx,
    __half *grad_attn, __half *grad_v, __half *grad_table);

#ifdef ENABLE_BF16
void dot_prod_with_idx_forward_cuda_launcher_bf16(int N, int M, int h, int hdim, int n_max, const int L,
    const __nv_bfloat16 *q, const int *index_q, const int *index_q_offsets, const __nv_bfloat16 *k, const int *index_k,
    const __nv_bfloat16 *table_q, const __nv_bfloat16 *table_k, const char *rel_idx, __nv_bfloat16 *output);
void dot_prod_with_idx_backward_cuda_launcher_bf16(int N, int M, int h, int hdim, int n_max, const int L,
    const __nv_bfloat16 *grad_out, const __nv_bfloat16 *q, const int *index_q_offsets, const __nv_bfloat16 *k,
    const int *index_k_offsets, const int *index_k, const __nv_bfloat16 *table_q, const __nv_bfloat16 *table_k,
    const char *rel_idx, __nv_bfloat16 *grad_q, __nv_bfloat16 *grad_k, __nv_bfloat16 *grad_table_q,
    __nv_bfloat16 *grad_table_k);

void dot_prod_with_idx_all_forward_cuda_launcher_bf16(int N, int M, int h, int hdim, int n_max, const int L,
    const __nv_bfloat16 *q, const int *index_q, const int *index_q_offsets, const __nv_bfloat16 *k, const int *index_k,
    const __nv_bfloat16 *table_q, const __nv_bfloat16 *table_k, const char *rel_idx, __nv_bfloat16 *output);
void dot_prod_with_idx_all_backward_cuda_launcher_bf16(int N, int M, int h, int hdim, int n_max, const int L,
    const __nv_bfloat16 *grad_out, const __nv_bfloat16 *q, const int *index_q_offsets, const __nv_bfloat16 *k,
    const int *index_k_offsets, const int *index_k, const __nv_bfloat16 *table_q, const __nv_bfloat16 *table_k,
    const char *rel_idx, __nv_bfloat16 *grad_q, __nv_bfloat16 *grad_k, __nv_bfloat16 *grad_table_q,
    __nv_bfloat16 *grad_table_k);

void attention_step2_with_rel_pos_value_forward_cuda_launcher_bf16(int N, int M, int h, int hdim, int n_max,
    const __nv_bfloat16 *attn, const __nv_bfloat16 *v, const int *index0_offsets, const int *index1,
    const __nv_bfloat16 *table, const char *rel_idx, __nv_bfloat16 *output);
void attention_step2_with_rel_pos_value_backward_cuda_launcher_bf16(int N, int M, int h, int hdim, int L,
    int n_max, const __nv_bfloat16 *grad_out, const int *index0, const int *index0_offsets, const int *index1,
    const int *index1_offsets, const __nv_bfloat16 *attn, const __nv_bfloat16 *v, const __nv_bfloat16 *table,
    const char *rel_idx, __nv_bfloat16 *grad_attn, __nv_bfloat16 *grad_v, __nv_bfloat16 *grad_table);
#endif

#ifdef __cplusplus
}
#endif
#endif
