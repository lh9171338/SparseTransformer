#include <torch/extension.h>
#include "relative_pos_encoding_cuda_kernel.h"

void dot_prod_with_idx_forward_cuda(int N, int M, int h, int hdim, int n_max, const int L, at::Tensor q_tensor,
    at::Tensor index_q_tensor, at::Tensor index_q_offsets_tensor, at::Tensor k_tensor, at::Tensor index_k_tensor,
    at::Tensor table_q_tensor, at::Tensor table_k_tensor, at::Tensor rel_idx_tensor, at::Tensor output_tensor)
{
    const int *index_q = index_q_tensor.data_ptr<int>();
    const int *index_q_offsets = index_q_offsets_tensor.data_ptr<int>();
    const int *index_k = index_k_tensor.data_ptr<int>();
    const char* rel_idx = static_cast<const char*>(rel_idx_tensor.data_ptr());
    if (q_tensor.scalar_type() == at::kHalf) {
        using T = __half;
        const T *q = static_cast<T*>(q_tensor.data_ptr());
        const T *k = static_cast<T*>(k_tensor.data_ptr());
        const T *table_q = static_cast<T*>(table_q_tensor.data_ptr());
        const T *table_k = static_cast<T*>(table_k_tensor.data_ptr());
        T *output = static_cast<T*>(output_tensor.data_ptr());
        dot_prod_with_idx_forward_cuda_launcher_fp16(N, M, h, hdim, n_max, L, q, index_q, index_q_offsets, k,
            index_k, table_q, table_k, rel_idx, output);
#ifdef ENABLE_BF16
    } else if (q_tensor.scalar_type() == at::kBFloat16) {
        using T = __nv_bfloat16;
        const T *q = static_cast<T*>(q_tensor.data_ptr());
        const T *k = static_cast<T*>(k_tensor.data_ptr());
        const T *table_q = static_cast<T*>(table_q_tensor.data_ptr());
        const T *table_k = static_cast<T*>(table_k_tensor.data_ptr());
        T *output = static_cast<T*>(output_tensor.data_ptr());
        dot_prod_with_idx_forward_cuda_launcher_bf16(N, M, h, hdim, n_max, L, q, index_q, index_q_offsets, k,
            index_k, table_q, table_k, rel_idx, output);
#endif
    } else {
        using T = float;
        const T *q = static_cast<T*>(q_tensor.data_ptr());
        const T *k = static_cast<T*>(k_tensor.data_ptr());
        const T *table_q = static_cast<T*>(table_q_tensor.data_ptr());
        const T *table_k = static_cast<T*>(table_k_tensor.data_ptr());
        T *output = static_cast<T*>(output_tensor.data_ptr());
        dot_prod_with_idx_forward_cuda_launcher_fp32(N, M, h, hdim, n_max, L, q, index_q, index_q_offsets, k,
            index_k, table_q, table_k, rel_idx, output);
    }
}

void dot_prod_with_idx_backward_cuda(int N, int M, int h, int hdim, int n_max, const int L, at::Tensor grad_out_tensor,
    at::Tensor q_tensor, at::Tensor index_q_offsets_tensor, at::Tensor k_tensor, at::Tensor index_k_offsets_tensor,
    at::Tensor index_k_tensor, at::Tensor table_q_tensor, at::Tensor table_k_tensor, at::Tensor rel_idx_tensor,
    at::Tensor grad_q_tensor, at::Tensor grad_k_tensor, at::Tensor grad_table_q_tensor, at::Tensor grad_table_k_tensor)
{
    const int *index_q_offsets = index_q_offsets_tensor.data_ptr<int>();
    const int *index_k_offsets = index_k_offsets_tensor.data_ptr<int>();
    const int *index_k = index_k_tensor.data_ptr<int>();
    const char* rel_idx = static_cast<const char*>(rel_idx_tensor.data_ptr());
    if (grad_out_tensor.scalar_type() == at::kHalf) {
        using T = __half;
        const T *grad_out = static_cast<T*>(grad_out_tensor.data_ptr());
        const T *q = static_cast<T*>(q_tensor.data_ptr());
        const T *k = static_cast<T*>(k_tensor.data_ptr());
        const T *table_q = static_cast<T*>(table_q_tensor.data_ptr());
        const T *table_k = static_cast<T*>(table_k_tensor.data_ptr());
        T *grad_q = static_cast<T*>(grad_q_tensor.data_ptr());
        T *grad_k = static_cast<T*>(grad_k_tensor.data_ptr());
        T *grad_table_q = static_cast<T*>(grad_table_q_tensor.data_ptr());
        T *grad_table_k = static_cast<T*>(grad_table_k_tensor.data_ptr());
        dot_prod_with_idx_backward_cuda_launcher_fp16(N, M, h, hdim, n_max, L, grad_out, q, index_q_offsets, k,
            index_k_offsets, index_k, table_q, table_k, rel_idx, grad_q, grad_k, grad_table_q, grad_table_k);
#ifdef ENABLE_BF16
    } else if (q_tensor.scalar_type() == at::kBFloat16) {
        using T = __nv_bfloat16;
        const T *grad_out = static_cast<T*>(grad_out_tensor.data_ptr());
        const T *q = static_cast<T*>(q_tensor.data_ptr());
        const T *k = static_cast<T*>(k_tensor.data_ptr());
        const T *table_q = static_cast<T*>(table_q_tensor.data_ptr());
        const T *table_k = static_cast<T*>(table_k_tensor.data_ptr());
        T *grad_q = static_cast<T*>(grad_q_tensor.data_ptr());
        T *grad_k = static_cast<T*>(grad_k_tensor.data_ptr());
        T *grad_table_q = static_cast<T*>(grad_table_q_tensor.data_ptr());
        T *grad_table_k = static_cast<T*>(grad_table_k_tensor.data_ptr());
        dot_prod_with_idx_backward_cuda_launcher_bf16(N, M, h, hdim, n_max, L, grad_out, q, index_q_offsets, k,
            index_k_offsets, index_k, table_q, table_k, rel_idx, grad_q, grad_k, grad_table_q, grad_table_k);
#endif
    } else {
        using T = float;
        const T *grad_out = static_cast<T*>(grad_out_tensor.data_ptr());
        const T *q = static_cast<T*>(q_tensor.data_ptr());
        const T *k = static_cast<T*>(k_tensor.data_ptr());
        const T *table_q = static_cast<T*>(table_q_tensor.data_ptr());
        const T *table_k = static_cast<T*>(table_k_tensor.data_ptr());
        T *grad_q = static_cast<T*>(grad_q_tensor.data_ptr());
        T *grad_k = static_cast<T*>(grad_k_tensor.data_ptr());
        T *grad_table_q = static_cast<T*>(grad_table_q_tensor.data_ptr());
        T *grad_table_k = static_cast<T*>(grad_table_k_tensor.data_ptr());
        dot_prod_with_idx_backward_cuda_launcher_fp32(N, M, h, hdim, n_max, L, grad_out, q, index_q_offsets, k,
            index_k_offsets, index_k, table_q, table_k, rel_idx, grad_q, grad_k, grad_table_q, grad_table_k);
    }
}

void dot_prod_with_idx_all_forward_cuda(int N, int M, int h, int hdim, int n_max, const int L, at::Tensor q_tensor,
    at::Tensor index_q_tensor, at::Tensor index_q_offsets_tensor, at::Tensor k_tensor, at::Tensor index_k_tensor,
    at::Tensor table_q_tensor, at::Tensor table_k_tensor, at::Tensor rel_idx_tensor, at::Tensor output_tensor)
{
    const int *index_q = index_q_tensor.data_ptr<int>();
    const int *index_q_offsets = index_q_offsets_tensor.data_ptr<int>();
    const int *index_k = index_k_tensor.data_ptr<int>();
    const char* rel_idx = static_cast<const char*>(rel_idx_tensor.data_ptr());
    if (q_tensor.scalar_type() == at::kHalf) {
        using T = __half;
        const T *q = static_cast<T*>(q_tensor.data_ptr());
        const T *k = static_cast<T*>(k_tensor.data_ptr());
        const T *table_q = static_cast<T*>(table_q_tensor.data_ptr());
        const T *table_k = static_cast<T*>(table_k_tensor.data_ptr());
        T *output = static_cast<T*>(output_tensor.data_ptr());
        dot_prod_with_idx_all_forward_cuda_launcher_fp16(N, M, h, hdim, n_max, L, q, index_q, index_q_offsets,
            k, index_k, table_q, table_k, rel_idx, output);
#ifdef ENABLE_BF16
    } else if (q_tensor.scalar_type() == at::kBFloat16) {
        using T = __nv_bfloat16;
        const T *q = static_cast<T*>(q_tensor.data_ptr());
        const T *k = static_cast<T*>(k_tensor.data_ptr());
        const T *table_q = static_cast<T*>(table_q_tensor.data_ptr());
        const T *table_k = static_cast<T*>(table_k_tensor.data_ptr());
        T *output = static_cast<T*>(output_tensor.data_ptr());
        dot_prod_with_idx_all_forward_cuda_launcher_bf16(N, M, h, hdim, n_max, L, q, index_q, index_q_offsets,
            k, index_k, table_q, table_k, rel_idx, output);
#endif
    } else {
        using T = float;
        const T *q = static_cast<T*>(q_tensor.data_ptr());
        const T *k = static_cast<T*>(k_tensor.data_ptr());
        const T *table_q = static_cast<T*>(table_q_tensor.data_ptr());
        const T *table_k = static_cast<T*>(table_k_tensor.data_ptr());
        T *output = static_cast<T*>(output_tensor.data_ptr());
        dot_prod_with_idx_all_forward_cuda_launcher_fp32(N, M, h, hdim, n_max, L, q, index_q, index_q_offsets,
            k, index_k, table_q, table_k, rel_idx, output);
    }
}

void dot_prod_with_idx_all_backward_cuda(int N, int M, int h, int hdim, int n_max, const int L, at::Tensor grad_out_tensor,
    at::Tensor q_tensor, at::Tensor index_q_offsets_tensor, at::Tensor k_tensor, at::Tensor index_k_offsets_tensor,
    at::Tensor index_k_tensor, at::Tensor table_q_tensor, at::Tensor table_k_tensor, at::Tensor rel_idx_tensor,
    at::Tensor grad_q_tensor, at::Tensor grad_k_tensor, at::Tensor grad_table_q_tensor, at::Tensor grad_table_k_tensor)
{
    const int *index_q_offsets = index_q_offsets_tensor.data_ptr<int>();
    const int *index_k_offsets = index_k_offsets_tensor.data_ptr<int>();
    const int *index_k = index_k_tensor.data_ptr<int>();
    const char* rel_idx = static_cast<const char*>(rel_idx_tensor.data_ptr());
    if (grad_out_tensor.scalar_type() == at::kHalf) {
        using T = __half;
        const T *grad_out = static_cast<T*>(grad_out_tensor.data_ptr());
        const T *q = static_cast<T*>(q_tensor.data_ptr());
        const T *k = static_cast<T*>(k_tensor.data_ptr());
        const T *table_q = static_cast<T*>(table_q_tensor.data_ptr());
        const T *table_k = static_cast<T*>(table_k_tensor.data_ptr());
        T *grad_q = static_cast<T*>(grad_q_tensor.data_ptr());
        T *grad_k = static_cast<T*>(grad_k_tensor.data_ptr());
        T *grad_table_q = static_cast<T*>(grad_table_q_tensor.data_ptr());
        T *grad_table_k = static_cast<T*>(grad_table_k_tensor.data_ptr());
        dot_prod_with_idx_all_backward_cuda_launcher_fp16(N, M, h, hdim, n_max, L, grad_out, q, index_q_offsets, k,
            index_k_offsets, index_k, table_q, table_k, rel_idx, grad_q, grad_k, grad_table_q, grad_table_k);
#ifdef ENABLE_BF16
    } else if (q_tensor.scalar_type() == at::kBFloat16) {
        using T = __nv_bfloat16;
        const T *grad_out = static_cast<T*>(grad_out_tensor.data_ptr());
        const T *q = static_cast<T*>(q_tensor.data_ptr());
        const T *k = static_cast<T*>(k_tensor.data_ptr());
        const T *table_q = static_cast<T*>(table_q_tensor.data_ptr());
        const T *table_k = static_cast<T*>(table_k_tensor.data_ptr());
        T *grad_q = static_cast<T*>(grad_q_tensor.data_ptr());
        T *grad_k = static_cast<T*>(grad_k_tensor.data_ptr());
        T *grad_table_q = static_cast<T*>(grad_table_q_tensor.data_ptr());
        T *grad_table_k = static_cast<T*>(grad_table_k_tensor.data_ptr());
        dot_prod_with_idx_all_backward_cuda_launcher_bf16(N, M, h, hdim, n_max, L, grad_out, q, index_q_offsets, k,
            index_k_offsets, index_k, table_q, table_k, rel_idx, grad_q, grad_k, grad_table_q, grad_table_k);
#endif
    } else {
        using T = float;
        const T *grad_out = static_cast<T*>(grad_out_tensor.data_ptr());
        const T *q = static_cast<T*>(q_tensor.data_ptr());
        const T *k = static_cast<T*>(k_tensor.data_ptr());
        const T *table_q = static_cast<T*>(table_q_tensor.data_ptr());
        const T *table_k = static_cast<T*>(table_k_tensor.data_ptr());
        T *grad_q = static_cast<T*>(grad_q_tensor.data_ptr());
        T *grad_k = static_cast<T*>(grad_k_tensor.data_ptr());
        T *grad_table_q = static_cast<T*>(grad_table_q_tensor.data_ptr());
        T *grad_table_k = static_cast<T*>(grad_table_k_tensor.data_ptr());
        dot_prod_with_idx_all_backward_cuda_launcher_fp32(N, M, h, hdim, n_max, L, grad_out, q, index_q_offsets, k,
            index_k_offsets, index_k, table_q, table_k, rel_idx, grad_q, grad_k, grad_table_q, grad_table_k);
    }
}

void attention_step2_with_rel_pos_value_forward_cuda(int N, int M, int h, int hdim, int n_max, at::Tensor attn_tensor,
    at::Tensor v_tensor, at::Tensor index0_offsets_tensor, at::Tensor index1_tensor, at::Tensor table_tensor,
    at::Tensor rel_idx_tensor, at::Tensor output_tensor)
{
    const int *index0_offsets = index0_offsets_tensor.data_ptr<int>();
    const int *index1 = index1_tensor.data_ptr<int>();
    const char* rel_idx = static_cast<const char*>(rel_idx_tensor.data_ptr());
    if (attn_tensor.scalar_type() == at::kHalf) {
        using T = __half;
        const T *attn = static_cast<T*>(attn_tensor.data_ptr());
        const T *v = static_cast<T*>(v_tensor.data_ptr());
        const T *table = static_cast<T*>(table_tensor.data_ptr());
        T *output = static_cast<T*>(output_tensor.data_ptr());
        attention_step2_with_rel_pos_value_forward_cuda_launcher_fp16(N, M, h, hdim, n_max, attn, v,
            index0_offsets, index1, table, rel_idx, output);
#ifdef ENABLE_BF16
    } else if (attn_tensor.scalar_type() == at::kBFloat16) {
        using T = __nv_bfloat16;
        const T *attn = static_cast<T*>(attn_tensor.data_ptr());
        const T *v = static_cast<T*>(v_tensor.data_ptr());
        const T *table = static_cast<T*>(table_tensor.data_ptr());
        T *output = static_cast<T*>(output_tensor.data_ptr());
        attention_step2_with_rel_pos_value_forward_cuda_launcher_bf16(N, M, h, hdim, n_max, attn, v,
            index0_offsets, index1, table, rel_idx, output);
#endif
    } else {
        using T = float;
        const T *attn = static_cast<T*>(attn_tensor.data_ptr());
        const T *v = static_cast<T*>(v_tensor.data_ptr());
        const T *table = static_cast<T*>(table_tensor.data_ptr());
        T *output = static_cast<T*>(output_tensor.data_ptr());
        attention_step2_with_rel_pos_value_forward_cuda_launcher_fp32(N, M, h, hdim, n_max, attn, v,
            index0_offsets, index1, table, rel_idx, output);
    }
}

void attention_step2_with_rel_pos_value_backward_cuda(int N, int M, int h, int hdim, int L, int n_max,
    at::Tensor grad_out_tensor, at::Tensor index0_tensor, at::Tensor index0_offsets_tensor, at::Tensor index1_tensor,
    at::Tensor index1_offsets_tensor, at::Tensor attn_tensor, at::Tensor v_tensor, at::Tensor table_tensor,
    at::Tensor rel_idx_tensor, at::Tensor grad_attn_tensor, at::Tensor grad_v_tensor, at::Tensor grad_table_tensor)
{
    const int *index0 = index0_tensor.data_ptr<int>();
    const int *index0_offsets = index0_offsets_tensor.data_ptr<int>();
    const int *index1 = index1_tensor.data_ptr<int>();
    const int *index1_offsets = index1_offsets_tensor.data_ptr<int>();
    const char* rel_idx = static_cast<const char*>(rel_idx_tensor.data_ptr());
    if (grad_out_tensor.scalar_type() == at::kHalf) {
        using T = __half;
        const T *grad_out = static_cast<T*>(grad_out_tensor.data_ptr());
        const T *attn = static_cast<T*>(attn_tensor.data_ptr());
        const T *v = static_cast<T*>(v_tensor.data_ptr());
        const T *table = static_cast<T*>(table_tensor.data_ptr());
        T *grad_attn = static_cast<T*>(grad_attn_tensor.data_ptr());
        T *grad_v = static_cast<T*>(grad_v_tensor.data_ptr());
        T *grad_table = static_cast<T*>(grad_table_tensor.data_ptr());
        attention_step2_with_rel_pos_value_backward_cuda_launcher_fp16(N, M, h, hdim, L, n_max, grad_out,
            index0, index0_offsets, index1, index1_offsets, attn, v, table, rel_idx, grad_attn, grad_v, grad_table);
#ifdef ENABLE_BF16
    } else if (grad_out_tensor.scalar_type() == at::kBFloat16) {
        using T = __nv_bfloat16;
        const T *grad_out = static_cast<T*>(grad_out_tensor.data_ptr());
        const T *attn = static_cast<T*>(attn_tensor.data_ptr());
        const T *v = static_cast<T*>(v_tensor.data_ptr());
        const T *table = static_cast<T*>(table_tensor.data_ptr());
        T *grad_attn = static_cast<T*>(grad_attn_tensor.data_ptr());
        T *grad_v = static_cast<T*>(grad_v_tensor.data_ptr());
        T *grad_table = static_cast<T*>(grad_table_tensor.data_ptr());
        attention_step2_with_rel_pos_value_backward_cuda_launcher_bf16(N, M, h, hdim, L, n_max, grad_out,
            index0, index0_offsets, index1, index1_offsets, attn, v, table, rel_idx, grad_attn, grad_v, grad_table);
#endif
    } else {
        using T = float;
        const T *grad_out = static_cast<T*>(grad_out_tensor.data_ptr());
        const T *attn = static_cast<T*>(attn_tensor.data_ptr());
        const T *v = static_cast<T*>(v_tensor.data_ptr());
        const T *table = static_cast<T*>(table_tensor.data_ptr());
        T *grad_attn = static_cast<T*>(grad_attn_tensor.data_ptr());
        T *grad_v = static_cast<T*>(grad_v_tensor.data_ptr());
        T *grad_table = static_cast<T*>(grad_table_tensor.data_ptr());
        attention_step2_with_rel_pos_value_backward_cuda_launcher_fp32(N, M, h, hdim, L, n_max, grad_out,
            index0, index0_offsets, index1, index1_offsets, attn, v, table, rel_idx, grad_attn, grad_v, grad_table);
    }
}
