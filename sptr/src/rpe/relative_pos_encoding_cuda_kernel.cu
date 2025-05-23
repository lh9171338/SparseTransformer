#include "../cuda_utils.h"
#include "relative_pos_encoding_cuda_kernel.h"

template <typename T>
__global__ void dot_prod_with_idx_forward_cuda_kernel( // M, h, hdim
    int N, int M, int h, int d, int L, const T *q, const int *index_q, const int *index_q_offsets, const T *k, const int *index_k,
    const T *table_q, const T *table_k, const char *rel_idx, T *output) {
    // input: q: (h, hdim, N), k: (h, hdim, N), index_q: (M), index_k: (M), table: (h, hdim, 3, L), rel_idx: (3, M), output: (h, M)

    int h_idx = blockIdx.y;
    int m_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(m_idx < M){
        int q_idx = index_q[m_idx];
        int k_idx = index_k[m_idx];
        char rel_idx1 = rel_idx[m_idx];
        char rel_idx2 = rel_idx[M + m_idx];
        char rel_idx3 = rel_idx[M*2 + m_idx];
        T s = float2type<T>(0.0f);
        for(int i = 0; i < d; i++){
            T q_scalar = q[h_idx*d*N + i*N + q_idx];
            T k_scalar = k[h_idx*d*N + i*N + k_idx];
            s += q_scalar * (table_q[h_idx*d*3*L + i*3*L + rel_idx1] + table_q[h_idx*d*3*L + i*3*L + L + rel_idx2] + table_q[h_idx*d*3*L + i*3*L + 2*L + rel_idx3]) +
                k_scalar * (table_k[h_idx*d*3*L + i*3*L + rel_idx1] + table_k[h_idx*d*3*L + i*3*L + L + rel_idx2] + table_k[h_idx*d*3*L + i*3*L + 2*L + rel_idx3]);
        }
        output[h_idx*M + m_idx] = s;
    }
}

void dot_prod_with_idx_forward_cuda_launcher_fp32(int N, int M, int h, int hdim, int n_max, const int L, const float *q,
    const int *index_q, const int *index_q_offsets, const float *k, const int *index_k, const float *table_q, const float *table_k,
    const char *rel_idx, float *output)
{
    // input: q: (N, h, hdim), index: (M), table: (L, h, hdim, 3), rel_idx: (M, 3)

    unsigned int n_threads = 512;
    dim3 blocks((M+n_threads-1)/n_threads, h);

    dot_prod_with_idx_forward_cuda_kernel<<<blocks, n_threads, 0>>>(N, M, h, hdim, L, q, index_q, index_q_offsets, k, index_k, table_q, table_k, rel_idx, output);
}

void dot_prod_with_idx_forward_cuda_launcher_fp16(int N, int M, int h, int hdim, int n_max, const int L, const __half *q,
    const int *index_q, const int *index_q_offsets, const __half *k, const int *index_k, const __half *table_q, const __half *table_k,
    const char *rel_idx, __half *output)
{
    // input: q: (N, h, hdim), index: (M), table: (L, h, hdim, 3), rel_idx: (M, 3)

    unsigned int n_threads = 512;
    dim3 blocks((M+n_threads-1)/n_threads, h);

    dot_prod_with_idx_forward_cuda_kernel<<<blocks, n_threads, 0>>>(N, M, h, hdim, L, q, index_q, index_q_offsets, k, index_k, table_q, table_k, rel_idx, output);
}

#ifdef ENABLE_BF16
void dot_prod_with_idx_forward_cuda_launcher_bf16(int N, int M, int h, int hdim, int n_max, const int L, const __nv_bfloat16 *q,
    const int *index_q, const int *index_q_offsets, const __nv_bfloat16 *k, const int *index_k, const __nv_bfloat16 *table_q, const __nv_bfloat16 *table_k,
    const char *rel_idx, __nv_bfloat16 *output)
{
    // input: q: (N, h, hdim), index: (M), table: (L, h, hdim, 3), rel_idx: (M, 3)

    unsigned int n_threads = 512;
    dim3 blocks((M+n_threads-1)/n_threads, h);

    dot_prod_with_idx_forward_cuda_kernel<<<blocks, n_threads, 0>>>(N, M, h, hdim, L, q, index_q, index_q_offsets, k, index_k, table_q, table_k, rel_idx, output);
}
#endif

template <typename T>
__global__ void dot_prod_with_idx_backward_cuda_kernel( // M, h, hdim
    int N, int M, int h, int d, const int L, const T *grad_out, const T *q, const int *index_q_offsets,
    const T *k, const int *index_k_offsets, const int *index_k, const T *table_q, const T *table_k,
    const char *rel_idx, T *grad_q, T *grad_k, T *grad_table_q, T *grad_table_k) {
    // Note: L <= 50
    // Input: q: [N, h, hdim], k: [N, h, hdim], index_k: [M, ], table_q: [L, 3, h, hdim], table_k: [L, 3, h, hdim], rel_idx: [M, 3], grad_out: [M, h]
    // Return: grad_q: [N, h, hdim], grad_k: [N, h, hdim], grad_table_q: [L, 3, h, hdim], grad_table_k: [L, 3, h, hdim]

    int q_idx = blockIdx.x;
    int t_idx = threadIdx.x;
    int d_idx = threadIdx.y;
    int n_h = blockDim.x;
    int h_idx = blockIdx.y * n_h + t_idx;
    int C = h*d;

    int start = index_q_offsets[q_idx];
    int end = index_q_offsets[q_idx+1];
    int n = end - start;

    T grad_q_val = float2type<T>(0.0f);
    T grad_table_q_reg[50][3] = {float2type<T>(0.0f)};
    for(int i = 0; i < n; i ++){
        int start_i = start + i;
        char rel_idx1 = rel_idx[start_i*3], rel_idx2 = rel_idx[start_i*3 + 1], rel_idx3 = rel_idx[start_i*3 + 2];
        T grad_out_val = grad_out[start_i*h + h_idx];
        grad_q_val += (table_q[rel_idx1*3*C + h_idx*d + d_idx] + table_q[rel_idx2*3*C + C + h_idx*d + d_idx] + table_q[rel_idx3*3*C + 2*C + h_idx*d + d_idx]) * grad_out_val;

        T grad_table_q_val = q[q_idx*C + h_idx*d + d_idx] * grad_out_val;
        grad_table_q_reg[rel_idx1][0] += grad_table_q_val;
        grad_table_q_reg[rel_idx2][1] += grad_table_q_val;
        grad_table_q_reg[rel_idx3][2] += grad_table_q_val;
    }

    grad_q[q_idx*C + h_idx*d + d_idx] = grad_q_val;
    for(int i = 0; i < L*3; i++){
        atomicAdd(grad_table_q + i*C + h_idx*d + d_idx, grad_table_q_reg[i/3][i%3]);
    }

    int start_k = index_k_offsets[q_idx];
    T grad_k_val = float2type<T>(0.0f);
    T grad_table_k_reg[50][3] = {float2type<T>(0.0f)};
    for(int i = 0; i < n; i ++){
        int start_i = start_k + i*n;
        char rel_idx1 = rel_idx[start_i*3], rel_idx2 = rel_idx[start_i*3 + 1], rel_idx3 = rel_idx[start_i*3 + 2];
        T grad_out_val = grad_out[start_i*h + h_idx];
        grad_k_val += (table_k[rel_idx1*3*C + h_idx*d + d_idx] + table_k[rel_idx2*3*C + C + h_idx*d + d_idx] + table_k[rel_idx3*3*C + 2*C + h_idx*d + d_idx]) * grad_out_val;

        T grad_table_k_val = k[q_idx*C + h_idx*d + d_idx] * grad_out_val;
        grad_table_k_reg[rel_idx1][0] += grad_table_k_val;
        grad_table_k_reg[rel_idx2][1] += grad_table_k_val;
        grad_table_k_reg[rel_idx3][2] += grad_table_k_val;
    }

    grad_k[q_idx*C + h_idx*d + d_idx] = grad_k_val;
    for(int i = 0; i < L*3; i++){
        atomicAdd(grad_table_k + i*C + h_idx*d + d_idx, grad_table_k_reg[i/3][i%3]);
    }
}

void dot_prod_with_idx_backward_cuda_launcher_fp32(int N, int M, int h, int hdim, int n_max, const int L,
    const float *grad_out, const float *q, const int *index_q_offsets, const float *k, const int *index_k_offsets, const int *index_k,
    const float *table_q, const float *table_k, const char *rel_idx,
    float *grad_q, float *grad_k, float *grad_table_q, float *grad_table_k)
{   //******* assert 512 % hdim == 0 *******
    // input: grad_out: (M, h), output: grad_q: (N, h, hdim), grad_table: (L, h, hdim, 3)

    unsigned int n_h = h * hdim > 512 ? 512 / hdim : h;
    dim3 blocks(N, h/n_h);
    dim3 threads(n_h, hdim);

    dot_prod_with_idx_backward_cuda_kernel<<<blocks, threads, 0>>>(N, M, h, hdim, L, grad_out, q, index_q_offsets, k, index_k_offsets, index_k, table_q, table_k, rel_idx, grad_q, grad_k, grad_table_q, grad_table_k);
}

void dot_prod_with_idx_backward_cuda_launcher_fp16(int N, int M, int h, int hdim, int n_max, const int L,
    const __half *grad_out, const __half *q, const int *index_q_offsets, const __half *k, const int *index_k_offsets, const int *index_k,
    const __half *table_q, const __half *table_k, const char *rel_idx,
    __half *grad_q, __half *grad_k, __half *grad_table_q, __half *grad_table_k)
{   //******* assert 512 % hdim == 0 *******
    // input: grad_out: (M, h), output: grad_q: (N, h, hdim), grad_table: (L, h, hdim, 3)

    unsigned int n_h = h * hdim > 512 ? 512 / hdim : h;
    dim3 blocks(N, h/n_h);
    dim3 threads(n_h, hdim);

    dot_prod_with_idx_backward_cuda_kernel<<<blocks, threads, 0>>>(N, M, h, hdim, L, grad_out, q, index_q_offsets, k, index_k_offsets, index_k, table_q, table_k, rel_idx, grad_q, grad_k, grad_table_q, grad_table_k);
}

#ifdef ENABLE_BF16
void dot_prod_with_idx_backward_cuda_launcher_bf16(int N, int M, int h, int hdim, int n_max, const int L,
    const __nv_bfloat16 *grad_out, const __nv_bfloat16 *q, const int *index_q_offsets, const __nv_bfloat16 *k, const int *index_k_offsets, const int *index_k,
    const __nv_bfloat16 *table_q, const __nv_bfloat16 *table_k, const char *rel_idx,
    __nv_bfloat16 *grad_q, __nv_bfloat16 *grad_k, __nv_bfloat16 *grad_table_q, __nv_bfloat16 *grad_table_k)
{   //******* assert 512 % hdim == 0 *******
    // input: grad_out: (M, h), output: grad_q: (N, h, hdim), grad_table: (L, h, hdim, 3)

    unsigned int n_h = h * hdim > 512 ? 512 / hdim : h;
    dim3 blocks(N, h/n_h);
    dim3 threads(n_h, hdim);

    dot_prod_with_idx_backward_cuda_kernel<<<blocks, threads, 0>>>(N, M, h, hdim, L, grad_out, q, index_q_offsets, k, index_k_offsets, index_k, table_q, table_k, rel_idx, grad_q, grad_k, grad_table_q, grad_table_k);
}
#endif

template <typename T>
__global__ void dot_prod_with_idx_all_forward_cuda_kernel( // M, h, hdim
    int N, int M, int h, int d, int L, const T *q, const int *index_q, const int *index_q_offsets, const T *k, const int *index_k,
    const T *table_q, const T *table_k, const char *rel_idx, T *output) {
    // input: q: (h, hdim, N), k: (h, hdim, N), index_q: (M), index_k: (M), table: (h, hdim, 3, L), rel_idx: (3, M), output: (h, M)

    int h_idx = blockIdx.y;
    int m_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(m_idx < M){
        int q_idx = index_q[m_idx];
        int k_idx = index_k[m_idx];
        char rel_idx1 = rel_idx[m_idx];
        char rel_idx2 = rel_idx[M + m_idx];
        char rel_idx3 = rel_idx[M*2 + m_idx];
        T s = float2type<T>(0.0f);
        for(int i = 0; i < d; i++){
            T q_scalar = q[h_idx*d*N + i*N + q_idx];
            T k_scalar = k[h_idx*d*N + i*N + k_idx];
            s += q_scalar * (k_scalar + table_q[h_idx*d*3*L + i*3*L + rel_idx1] + table_q[h_idx*d*3*L + i*3*L + L + rel_idx2] + table_q[h_idx*d*3*L + i*3*L + 2*L + rel_idx3]) +
                k_scalar * (table_k[h_idx*d*3*L + i*3*L + rel_idx1] + table_k[h_idx*d*3*L + i*3*L + L + rel_idx2] + table_k[h_idx*d*3*L + i*3*L + 2*L + rel_idx3]);
        }
        output[h_idx*M + m_idx] = s;
    }
}

void dot_prod_with_idx_all_forward_cuda_launcher_fp32(int N, int M, int h, int hdim, int n_max, const int L, const float *q,
    const int *index_q, const int *index_q_offsets, const float *k, const int *index_k, const float *table_q, const float *table_k,
    const char *rel_idx, float *output)
{
    // input: q: (N, h, hdim), index: (M), table: (L, h, hdim, 3), rel_idx: (M, 3)
    unsigned int n_threads = 512;
    dim3 blocks((M+n_threads-1)/n_threads, h);
    dot_prod_with_idx_all_forward_cuda_kernel<<<blocks, n_threads, 0>>>(N, M, h, hdim, L, q, index_q, index_q_offsets, k, index_k, table_q, table_k, rel_idx, output);
}

void dot_prod_with_idx_all_forward_cuda_launcher_fp16(int N, int M, int h, int hdim, int n_max, const int L, const __half *q,
    const int *index_q, const int *index_q_offsets, const __half *k, const int *index_k, const __half *table_q, const __half *table_k,
    const char *rel_idx, __half *output)
{
    // input: q: (N, h, hdim), index: (M), table: (L, h, hdim, 3), rel_idx: (M, 3)
    unsigned int n_threads = 512;
    dim3 blocks((M+n_threads-1)/n_threads, h);
    dot_prod_with_idx_all_forward_cuda_kernel<<<blocks, n_threads, 0>>>(N, M, h, hdim, L, q, index_q, index_q_offsets, k, index_k, table_q, table_k, rel_idx, output);
}

#ifdef ENABLE_BF16
void dot_prod_with_idx_all_forward_cuda_launcher_bf16(int N, int M, int h, int hdim, int n_max, const int L, const __nv_bfloat16 *q,
    const int *index_q, const int *index_q_offsets, const __nv_bfloat16 *k, const int *index_k, const __nv_bfloat16 *table_q, const __nv_bfloat16 *table_k,
    const char *rel_idx, __nv_bfloat16 *output)
{
    // input: q: (N, h, hdim), index: (M), table: (L, h, hdim, 3), rel_idx: (M, 3)
    unsigned int n_threads = 512;
    dim3 blocks((M+n_threads-1)/n_threads, h);
    dot_prod_with_idx_all_forward_cuda_kernel<<<blocks, n_threads, 0>>>(N, M, h, hdim, L, q, index_q, index_q_offsets, k, index_k, table_q, table_k, rel_idx, output);
}
#endif

template <typename T>
__global__ void dot_prod_with_idx_all_backward_cuda_kernel( // M, h, hdim
    int N, int M, int h, int d, const int L, const T *grad_out, const T *q, const int *index_q_offsets,
    const T *k, const int *index_k_offsets, const int *index_k, const T *table_q, const T *table_k,
    const char *rel_idx, T *grad_q, T *grad_k, T *grad_table_q, T *grad_table_k) {
    // Note: L <= 50
    // Input: q: [N, h, hdim], k: [N, h, hdim], index_k: [M, ], table_q: [L, 3, h, hdim], table_k: [L, 3, h, hdim], rel_idx: [M, 3], grad_out: [M, h]
    // Return: grad_q: [N, h, hdim], grad_k: [N, h, hdim], grad_table_q: [L, 3, h, hdim], grad_table_k: [L, 3, h, hdim]

    int q_idx = blockIdx.x;
    int t_idx = threadIdx.x;
    int d_idx = threadIdx.y;
    int n_h = blockDim.x;
    int h_idx = blockIdx.y * n_h + t_idx;
    int C = h*d;

    int start = index_q_offsets[q_idx];
    int end = index_q_offsets[q_idx+1];
    int n = end - start;

    T q_scalar = q[q_idx*C + h_idx*d + d_idx];
    T k_scalar = k[q_idx*C + h_idx*d + d_idx];
    T grad_q_val = float2type<T>(0.0f);
    T grad_table_q_reg[50][3] = {float2type<T>(0.0f)};
    for(int i = 0; i < n; i ++){
        int start_i = start + i;
        char rel_idx1 = rel_idx[start_i*3], rel_idx2 = rel_idx[start_i*3 + 1], rel_idx3 = rel_idx[start_i*3 + 2];
        T grad_out_val = grad_out[start_i*h + h_idx];
        grad_q_val += (table_q[rel_idx1*3*C + h_idx*d + d_idx] + table_q[rel_idx2*3*C + C + h_idx*d + d_idx]
             + table_q[rel_idx3*3*C + 2*C + h_idx*d + d_idx] + k_scalar) * grad_out_val;

        T grad_table_q_val = q_scalar * grad_out_val;
        grad_table_q_reg[rel_idx1][0] += grad_table_q_val;
        grad_table_q_reg[rel_idx2][1] += grad_table_q_val;
        grad_table_q_reg[rel_idx3][2] += grad_table_q_val;
    }

    grad_q[q_idx*C + h_idx*d + d_idx] = grad_q_val;
    for(int i = 0; i < L*3; i++){
        atomicAdd(grad_table_q + i*C + h_idx*d + d_idx, grad_table_q_reg[i/3][i%3]);
    }

    int start_k = index_k_offsets[q_idx];
    T grad_k_val = float2type<T>(0.0f);
    T grad_table_k_reg[50][3] = {float2type<T>(0.0f)};
    for(int i = 0; i < n; i ++){
        int start_i = start_k + i*n;
        char rel_idx1 = rel_idx[start_i*3], rel_idx2 = rel_idx[start_i*3 + 1], rel_idx3 = rel_idx[start_i*3 + 2];
        T grad_out_val = grad_out[start_i*h + h_idx];
        grad_k_val += (table_k[rel_idx1*3*C + h_idx*d + d_idx] + table_k[rel_idx2*3*C + C + h_idx*d + d_idx]
             + table_k[rel_idx3*3*C + 2*C + h_idx*d + d_idx] + q_scalar) * grad_out_val;

        T grad_table_k_val = k_scalar * grad_out_val;
        grad_table_k_reg[rel_idx1][0] += grad_table_k_val;
        grad_table_k_reg[rel_idx2][1] += grad_table_k_val;
        grad_table_k_reg[rel_idx3][2] += grad_table_k_val;
    }

    grad_k[q_idx*C + h_idx*d + d_idx] = grad_k_val;
    for(int i = 0; i < L*3; i++){
        atomicAdd(grad_table_k + i*C + h_idx*d + d_idx, grad_table_k_reg[i/3][i%3]);
    }
}

void dot_prod_with_idx_all_backward_cuda_launcher_fp32(int N, int M, int h, int hdim, int n_max, const int L,
    const float *grad_out, const float *q, const int *index_q_offsets, const float *k, const int *index_k_offsets, const int *index_k,
    const float *table_q, const float *table_k, const char *rel_idx,
    float *grad_q, float *grad_k, float *grad_table_q, float *grad_table_k)
{   //******* assert 512 % hdim == 0 *******
    // input: grad_out: (M, h), output: grad_q: (N, h, hdim), grad_table: (L, h, hdim, 3)

    unsigned int n_h = h * hdim > 512 ? 512 / hdim : h;
    dim3 blocks(N, h/n_h);
    dim3 threads(n_h, hdim);

    dot_prod_with_idx_all_backward_cuda_kernel<<<blocks, threads, 0>>>(N, M, h, hdim, L, grad_out, q, index_q_offsets, k, index_k_offsets, index_k, table_q, table_k, rel_idx, grad_q, grad_k, grad_table_q, grad_table_k);
}

void dot_prod_with_idx_all_backward_cuda_launcher_fp16(int N, int M, int h, int hdim, int n_max, const int L,
    const __half *grad_out, const __half *q, const int *index_q_offsets, const __half *k, const int *index_k_offsets, const int *index_k,
    const __half *table_q, const __half *table_k, const char *rel_idx,
    __half *grad_q, __half *grad_k, __half *grad_table_q, __half *grad_table_k)
{   //******* assert 512 % hdim == 0 *******
    // input: grad_out: (M, h), output: grad_q: (N, h, hdim), grad_table: (L, h, hdim, 3)

    unsigned int n_h = h * hdim > 512 ? 512 / hdim : h;
    dim3 blocks(N, h/n_h);
    dim3 threads(n_h, hdim);

    dot_prod_with_idx_all_backward_cuda_kernel<<<blocks, threads, 0>>>(N, M, h, hdim, L, grad_out, q, index_q_offsets, k, index_k_offsets, index_k, table_q, table_k, rel_idx, grad_q, grad_k, grad_table_q, grad_table_k);
}

#ifdef ENABLE_BF16
void dot_prod_with_idx_all_backward_cuda_launcher_bf16(int N, int M, int h, int hdim, int n_max, const int L,
    const __nv_bfloat16 *grad_out, const __nv_bfloat16 *q, const int *index_q_offsets, const __nv_bfloat16 *k, const int *index_k_offsets, const int *index_k,
    const __nv_bfloat16 *table_q, const __nv_bfloat16 *table_k, const char *rel_idx,
    __nv_bfloat16 *grad_q, __nv_bfloat16 *grad_k, __nv_bfloat16 *grad_table_q, __nv_bfloat16 *grad_table_k)
{   //******* assert 512 % hdim == 0 *******
    // input: grad_out: (M, h), output: grad_q: (N, h, hdim), grad_table: (L, h, hdim, 3)

    unsigned int n_h = h * hdim > 512 ? 512 / hdim : h;
    dim3 blocks(N, h/n_h);
    dim3 threads(n_h, hdim);

    dot_prod_with_idx_all_backward_cuda_kernel<<<blocks, threads, 0>>>(N, M, h, hdim, L, grad_out, q, index_q_offsets, k, index_k_offsets, index_k, table_q, table_k, rel_idx, grad_q, grad_k, grad_table_q, grad_table_k);
}
#endif

template <typename T>
__global__ void attention_step2_with_rel_pos_value_forward_cuda_kernel( // M, h, hdim
    int N, int M, const int h, int d, const T *attn, const T *v,
    const int *index0_offsets, const int *index1, const T *table, const char *rel_idx, T *output) {
    // input: attn: (M, h), v: (N, h, hdim), index0: (M, ), index1: (M, ), table: (L, 3, h, hdim), rel_idx: (M, 3)

    int q_idx = blockIdx.x;
    int n_h = blockDim.y;
    int h_idx = blockIdx.y * n_h + threadIdx.y;
    int d_idx = threadIdx.x;

    int C = h*d;

    int start = index0_offsets[q_idx], end = index0_offsets[q_idx+1];
    int n = end - start;
    T sum = float2type<T>(0.0f);
    for(int i = 0; i < n; i++){
        int start_i = start + i;
        char rel_idx1 = rel_idx[start_i*3], rel_idx2 = rel_idx[start_i*3 + 1], rel_idx3 = rel_idx[start_i*3 + 2];
        int k_idx = index1[start_i];
        T v_val = v[k_idx*C + h_idx*d + d_idx];
        sum += attn[start_i*h + h_idx] * (v_val + table[rel_idx1*C*3 + h_idx*d + d_idx] + table[rel_idx2*C*3 + C + h_idx*d + d_idx] + table[rel_idx3*C*3 + 2*C + h_idx*d + d_idx]);
    }
    output[q_idx*C + h_idx*d + d_idx] = sum;
}

void attention_step2_with_rel_pos_value_forward_cuda_launcher_fp32(int N, int M, const int h, int hdim, int n_max, const float *attn, const float *v, const int *index0_offsets,
    const int *index1, const float *table, const char *rel_idx, float *output) {
    // input: attn: (M, h), v: (N, h, hdim), index0: (M, ), index1: (M, ), table: (L, 3, h, hdim), rel_idx: (M, 3)

    unsigned int n_h = h*hdim > 512 ? 512 / hdim : h;
    dim3 blocks(N, h/n_h);
    dim3 threads(hdim, n_h);

	attention_step2_with_rel_pos_value_forward_cuda_kernel<<<blocks, threads, 0>>>(N, M, h, hdim, attn, v, index0_offsets, index1, table, rel_idx, output);
}

void attention_step2_with_rel_pos_value_forward_cuda_launcher_fp16(int N, int M, const int h, int hdim, int n_max, const __half *attn, const __half *v, const int *index0_offsets,
    const int *index1, const __half *table, const char *rel_idx, __half *output) {
    // input: attn: (M, h), v: (N, h, hdim), index0: (M, ), index1: (M, ), table: (L, 3, h, hdim), rel_idx: (M, 3)

    unsigned int n_h = h*hdim > 512 ? 512 / hdim : h;
    dim3 blocks(N, h/n_h);
    dim3 threads(hdim, n_h);

	attention_step2_with_rel_pos_value_forward_cuda_kernel<<<blocks, threads, 0>>>(N, M, h, hdim, attn, v, index0_offsets, index1, table, rel_idx, output);
}

#ifdef ENABLE_BF16
void attention_step2_with_rel_pos_value_forward_cuda_launcher_bf16(int N, int M, const int h, int hdim, int n_max, const __nv_bfloat16 *attn, const __nv_bfloat16 *v, const int *index0_offsets,
    const int *index1, const __nv_bfloat16 *table, const char *rel_idx, __nv_bfloat16 *output) {
    // input: attn: (M, h), v: (N, h, hdim), index0: (M, ), index1: (M, ), table: (L, 3, h, hdim), rel_idx: (M, 3)

    unsigned int n_h = h*hdim > 512 ? 512 / hdim : h;
    dim3 blocks(N, h/n_h);
    dim3 threads(hdim, n_h);

	attention_step2_with_rel_pos_value_forward_cuda_kernel<<<blocks, threads, 0>>>(N, M, h, hdim, attn, v, index0_offsets, index1, table, rel_idx, output);
}
#endif

template <typename T>
__global__ void attention_step2_with_rel_pos_value_grad_v_table_backward_cuda_kernel( // M, h, hdim
    int N, int M, int h, int hdim, int L, const T *grad_out, const int *index0, const int *index0_offsets, const int *index1, const int *index1_offsets, const T *attn, const T *v, const T *table,
    const char *rel_idx, T *grad_v, T *grad_table) {
    // input: attn: (M, h), v: (h, hdim, N), index0: (M, ), index1: (M, ), table: (h, hdim, 3, L), rel_idx: (3, M)

    int q_idx = blockIdx.x;
    // int h_idx = blockIdx.y;
    // int n_idx = threadIdx.x;
    int n_h = blockDim.x;
    int h_idx = blockIdx.y * n_h + threadIdx.x;
    int d_idx = threadIdx.y;

    int C = h*hdim;

    int start = index0_offsets[q_idx], end = index0_offsets[q_idx+1];
    int n = end - start;
    int start_k = index1_offsets[q_idx];
    T grad_table_reg[50][3] = {float2type<T>(0.0f)};
    T grad_v_val = float2type<T>(0.0f);

    for(int i = 0; i < n; i ++){
        int start_i = start_k + i*n;
        char rel_idx1 = rel_idx[start_i], rel_idx2 = rel_idx[start_i + M], rel_idx3 = rel_idx[start_i + 2*M];
        int query_idx = index0[start_i];
        T grad_out_val = grad_out[query_idx*C + h_idx*hdim + d_idx];

        T grad_val = attn[start_i*h + h_idx] * grad_out_val;

        grad_table_reg[rel_idx1][0] += grad_val;
        grad_table_reg[rel_idx2][1] += grad_val;
        grad_table_reg[rel_idx3][2] += grad_val;

        grad_v_val += grad_val;
    }
    grad_v[q_idx*C + h_idx*hdim + d_idx] = grad_v_val;

    for(int i = 0; i < 3*L; i++){
        atomicAdd(grad_table + i*C + h_idx*hdim + d_idx, grad_table_reg[i/3][i%3]);
    }
}

template <typename T>
__global__ void attention_step2_with_rel_pos_value_grad_attn_backward_cuda_kernel( // M, h, hdim
    int N, int M, int h, int hdim, int L, const T *grad_out, const int *index0_offsets, const int *index1, const int *index1_offsets, const T *attn, const T *v, const T *table,
    const char *rel_idx, T *grad_attn) {
    // input: attn: (M, h), v: (h, hdim, N), index0: (M, ), index1: (M, ), table: (h, hdim, 3, L), rel_idx: (3, M)

    int q_idx = blockIdx.x;
    int h_idx = blockIdx.y;
    int t_idx = threadIdx.x;

    int d = hdim;
    int C = h*d;

    int start = index0_offsets[q_idx], end = index0_offsets[q_idx+1];
    int n = end - start;
    for(int n_idx = t_idx; n_idx < n; n_idx += blockDim.x){
        int start_i = start + n_idx;
        T grad_attn_val = float2type<T>(0.0f);
        for(int j = 0; j < hdim; j++){
            char rel_idx1 = rel_idx[start_i], rel_idx2 = rel_idx[start_i + M], rel_idx3 = rel_idx[start_i + M*2];
            int k_idx = index1[start_i];
            T grad_out_val = grad_out[q_idx*C + h_idx*hdim + j];
            grad_attn_val += grad_out_val * (table[h_idx*hdim*3*L + j*3*L + rel_idx1] + table[h_idx*hdim*3*L + j*3*L + L + rel_idx2] + table[h_idx*hdim*3*L + j*3*L + 2*L + rel_idx3] + v[h_idx*hdim*N + j*N + k_idx]);
        }
        grad_attn[start_i*h + h_idx] = grad_attn_val;
    }
}

void attention_step2_with_rel_pos_value_backward_cuda_launcher_fp32(int N, int M, int h, int hdim, int L, int n_max, const float *grad_out, const int *index0, const int *index0_offsets,
    const int *index1, const int *index1_offsets, const float *attn, const float *v, const float *table, const char *rel_idx, float *grad_attn, float *grad_v, float *grad_table) {
    // input: grad_output: (N, h, hdim)

    dim3 blocks;
    dim3 threads;

    unsigned int n_h = h*hdim > 512 ? 512 / hdim : h;
    blocks = dim3(N, h/n_h);
    threads = dim3(n_h, hdim);
    attention_step2_with_rel_pos_value_grad_v_table_backward_cuda_kernel<<<blocks, threads, 0>>>(N, M, h, hdim, L, grad_out, index0, index0_offsets, index1, index1_offsets, attn, v, table, rel_idx, grad_v, grad_table);

    dim3 blocks_2(N, h);
	unsigned int n_threads = opt_n_threads(n_max);
    n_threads = n_threads == n_max ? n_threads : n_threads * 2;
    n_threads = n_threads > 1024 ? 1024 : n_threads;

    attention_step2_with_rel_pos_value_grad_attn_backward_cuda_kernel<<<blocks_2, n_threads, 0>>>(N, M, h, hdim, L, grad_out, index0_offsets, index1, index1_offsets, attn, v, table, rel_idx, grad_attn);
}

void attention_step2_with_rel_pos_value_backward_cuda_launcher_fp16(int N, int M, int h, int hdim, int L, int n_max, const __half *grad_out, const int *index0, const int *index0_offsets,
    const int *index1, const int *index1_offsets, const __half *attn, const __half *v, const __half *table, const char *rel_idx, __half *grad_attn, __half *grad_v, __half *grad_table) {
    // input: grad_output: (N, h, hdim)

    dim3 blocks;
    dim3 threads;

    unsigned int n_h = h*hdim > 512 ? 512 / hdim : h;
    blocks = dim3(N, h/n_h);
    threads = dim3(n_h, hdim);
    attention_step2_with_rel_pos_value_grad_v_table_backward_cuda_kernel<<<blocks, threads, 0>>>(N, M, h, hdim, L, grad_out, index0, index0_offsets, index1, index1_offsets, attn, v, table, rel_idx, grad_v, grad_table);

    dim3 blocks_2(N, h);
	unsigned int n_threads = opt_n_threads(n_max);
    n_threads = n_threads == n_max ? n_threads : n_threads * 2;
    n_threads = n_threads > 1024 ? 1024 : n_threads;

    attention_step2_with_rel_pos_value_grad_attn_backward_cuda_kernel<<<blocks_2, n_threads, 0>>>(N, M, h, hdim, L, grad_out, index0_offsets, index1, index1_offsets, attn, v, table, rel_idx, grad_attn);
}

#ifdef ENABLE_BF16
void attention_step2_with_rel_pos_value_backward_cuda_launcher_bf16(int N, int M, int h, int hdim, int L, int n_max, const __nv_bfloat16 *grad_out, const int *index0, const int *index0_offsets,
    const int *index1, const int *index1_offsets, const __nv_bfloat16 *attn, const __nv_bfloat16 *v, const __nv_bfloat16 *table, const char *rel_idx, __nv_bfloat16 *grad_attn, __nv_bfloat16 *grad_v, __nv_bfloat16 *grad_table) {
    // input: grad_output: (N, h, hdim)

    dim3 blocks;
    dim3 threads;

    unsigned int n_h = h*hdim > 512 ? 512 / hdim : h;
    blocks = dim3(N, h/n_h);
    threads = dim3(n_h, hdim);
    attention_step2_with_rel_pos_value_grad_v_table_backward_cuda_kernel<<<blocks, threads, 0>>>(N, M, h, hdim, L, grad_out, index0, index0_offsets, index1, index1_offsets, attn, v, table, rel_idx, grad_v, grad_table);

    dim3 blocks_2(N, h);
	unsigned int n_threads = opt_n_threads(n_max);
    n_threads = n_threads == n_max ? n_threads : n_threads * 2;
    n_threads = n_threads > 1024 ? 1024 : n_threads;

    attention_step2_with_rel_pos_value_grad_attn_backward_cuda_kernel<<<blocks_2, n_threads, 0>>>(N, M, h, hdim, L, grad_out, index0_offsets, index1, index1_offsets, attn, v, table, rel_idx, grad_attn);
}
#endif
