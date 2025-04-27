#include <torch/extension.h>
#include "attention_cuda_kernel.h"

void attention_step1_forward_cuda(int N_q, int N_k, int M, int h, int hdim, const unsigned int n_max,
    at::Tensor q_tensor, at::Tensor k_tensor,
    at::Tensor index0_tensor, at::Tensor index1_tensor, at::Tensor attn_tensor)
{
    const int *index0 = index0_tensor.data_ptr<int>();
    const int *index1 = index1_tensor.data_ptr<int>();
    if (q_tensor.scalar_type() == at::kHalf) {
        using T = __half;
        const T* q = static_cast<T*>(q_tensor.data_ptr());
        const T* k = static_cast<T*>(k_tensor.data_ptr());
        T* attn = static_cast<T*>(attn_tensor.data_ptr());
        attention_step1_forward_cuda_launcher_fp16(N_q, N_k, M, h, hdim, n_max, q, k, index0, index1, attn);
#ifdef ENABLE_BF16
    } else if (q_tensor.scalar_type() == at::kBFloat16) {
        using T = __nv_bfloat16;
        const T* q = static_cast<T*>(q_tensor.data_ptr());
        const T* k = static_cast<T*>(k_tensor.data_ptr());
        T* attn = static_cast<T*>(attn_tensor.data_ptr());
        attention_step1_forward_cuda_launcher_bf16(N_q, N_k, M, h, hdim, n_max, q, k, index0, index1, attn);
#endif
    } else {
        using T = float;
        const T* q = static_cast<T*>(q_tensor.data_ptr());
        const T* k = static_cast<T*>(k_tensor.data_ptr());
        T* attn = static_cast<T*>(attn_tensor.data_ptr());
        attention_step1_forward_cuda_launcher_fp32(N_q, N_k, M, h, hdim, n_max, q, k, index0, index1, attn);
    }
}

void attention_step1_backward_cuda(int N, int M, int h, int hdim, const unsigned int n_max,
    at::Tensor grad_out_tensor, at::Tensor index0_tensor, at::Tensor index0_tensor_offsets,
    at::Tensor index1_tensor, at::Tensor index1_tensor_offsets, at::Tensor q_tensor, at::Tensor k_tensor,
    at::Tensor grad_q_tensor, at::Tensor grad_k_tensor)
{
    const int *index0 = index0_tensor.data_ptr<int>();
    const int *index0_offsets = index0_tensor_offsets.data_ptr<int>();
    const int *index1 = index1_tensor.data_ptr<int>();
    const int *index1_offsets = index1_tensor_offsets.data_ptr<int>();

    if (grad_out_tensor.scalar_type() == at::kHalf) {
        using T = __half;
        const T *grad_out = static_cast<T*>(grad_out_tensor.data_ptr());
        const T *q = static_cast<T*>(q_tensor.data_ptr());
        const T *k = static_cast<T*>(k_tensor.data_ptr());
        T *grad_q = static_cast<T*>(grad_q_tensor.data_ptr());
        T *grad_k = static_cast<T*>(grad_k_tensor.data_ptr());
        attention_step1_backward_cuda_launcher_fp16(N, M, h, hdim, n_max, grad_out, index0,
            index0_offsets, index1, index1_offsets, q, k, grad_q, grad_k);
#ifdef ENABLE_BF16
    } else if (grad_out_tensor.scalar_type() == at::kBFloat16) {
        using T = __nv_bfloat16;
        const T *grad_out = static_cast<T*>(grad_out_tensor.data_ptr());
        const T *q = static_cast<T*>(q_tensor.data_ptr());
        const T *k = static_cast<T*>(k_tensor.data_ptr());
        T *grad_q = static_cast<T*>(grad_q_tensor.data_ptr());
        T *grad_k = static_cast<T*>(grad_k_tensor.data_ptr());
        attention_step1_backward_cuda_launcher_bf16(N, M, h, hdim, n_max, grad_out, index0,
            index0_offsets, index1, index1_offsets, q, k, grad_q, grad_k);
#endif
    } else {
        using T = float;
        const T *grad_out = static_cast<T*>(grad_out_tensor.data_ptr());
        const T *q = static_cast<T*>(q_tensor.data_ptr());
        const T *k = static_cast<T*>(k_tensor.data_ptr());
        T *grad_q = static_cast<T*>(grad_q_tensor.data_ptr());
        T *grad_k = static_cast<T*>(grad_k_tensor.data_ptr());
        attention_step1_backward_cuda_launcher_fp32(N, M, h, hdim, n_max, grad_out, index0,
            index0_offsets, index1, index1_offsets, q, k, grad_q, grad_k);
    }
}

void attention_step2_forward_cuda(int N, int M, int h, int hdim, int n_max, at::Tensor attn_tensor,
    at::Tensor v_tensor, at::Tensor index0_offsets_tensor, at::Tensor index1_tensor, at::Tensor output_tensor)
{
    const int *index0_offsets = index0_offsets_tensor.data_ptr<int>();
    const int *index1 = index1_tensor.data_ptr<int>();

    if (attn_tensor.scalar_type() == at::kHalf) {
        using T = __half;
        const T *attn = static_cast<T*>(attn_tensor.data_ptr());
        const T *v = static_cast<T*>(v_tensor.data_ptr());
        T *output = static_cast<T*>(output_tensor.data_ptr());
        attention_step2_forward_cuda_launcher_fp16(N, M, h, hdim, n_max, attn, v, index0_offsets, index1, output);
#ifdef ENABLE_BF16
    } else if (attn_tensor.scalar_type() == at::kBFloat16) {
        using T = __nv_bfloat16;
        const T *attn = static_cast<T*>(attn_tensor.data_ptr());
        const T *v = static_cast<T*>(v_tensor.data_ptr());
        T *output = static_cast<T*>(output_tensor.data_ptr());
        attention_step2_forward_cuda_launcher_bf16(N, M, h, hdim, n_max, attn, v, index0_offsets, index1, output);
#endif
    } else {
        using T = float;
        const T *attn = static_cast<T*>(attn_tensor.data_ptr());
        const T *v = static_cast<T*>(v_tensor.data_ptr());
        T *output = static_cast<T*>(output_tensor.data_ptr());
        attention_step2_forward_cuda_launcher_fp32(N, M, h, hdim, n_max, attn, v, index0_offsets, index1, output);
    }
}

void attention_step2_backward_cuda(int N, int M, int h, int hdim, int n_max, at::Tensor grad_out_tensor,
    at::Tensor index0_tensor, at::Tensor index0_offsets_tensor, at::Tensor index1_tensor,
    at::Tensor index1_offsets_tensor, at::Tensor attn_tensor, at::Tensor v_tensor,
    at::Tensor grad_attn_tensor, at::Tensor grad_v_tensor)
{

    const int *index0 = index0_tensor.data_ptr<int>();
    const int *index0_offsets = index0_offsets_tensor.data_ptr<int>();
    const int *index1 = index1_tensor.data_ptr<int>();
    const int *index1_offsets = index1_offsets_tensor.data_ptr<int>();

    if (grad_out_tensor.scalar_type() == at::kHalf) {
        using T = __half;
        const T *grad_out = static_cast<const T*>(grad_out_tensor.data_ptr());
        const T *attn = static_cast<const T*>(attn_tensor.data_ptr());
        const T *v = static_cast<const T*>(v_tensor.data_ptr());
        T *grad_attn = static_cast<T*>(grad_attn_tensor.data_ptr());
        T *grad_v = static_cast<T*>(grad_v_tensor.data_ptr());
        attention_step2_backward_cuda_launcher_fp16(N, M, h, hdim, n_max, grad_out, index0, index0_offsets,
                index1, index1_offsets, attn, v, grad_attn, grad_v);
#ifdef ENABLE_BF16
    } else if (attn_tensor.scalar_type() == at::kBFloat16) {
        using T = __nv_bfloat16;
        const T *grad_out = static_cast<const T*>(grad_out_tensor.data_ptr());
        const T *attn = static_cast<const T*>(attn_tensor.data_ptr());
        const T *v = static_cast<const T*>(v_tensor.data_ptr());
        T *grad_attn = static_cast<T*>(grad_attn_tensor.data_ptr());
        T *grad_v = static_cast<T*>(grad_v_tensor.data_ptr());
        attention_step2_backward_cuda_launcher_bf16(N, M, h, hdim, n_max, grad_out, index0, index0_offsets,
                index1, index1_offsets, attn, v, grad_attn, grad_v);
#endif
    } else {
        using T = float;
        const T *grad_out = static_cast<const T*>(grad_out_tensor.data_ptr());
        const T *attn = static_cast<const T*>(attn_tensor.data_ptr());
        const T *v = static_cast<const T*>(v_tensor.data_ptr());
        T *grad_attn = static_cast<T*>(grad_attn_tensor.data_ptr());
        T *grad_v = static_cast<T*>(grad_v_tensor.data_ptr());
        attention_step2_backward_cuda_launcher_fp32(N, M, h, hdim, n_max, grad_out, index0, index0_offsets,
                index1, index1_offsets, attn, v, grad_attn, grad_v);
    }
}
