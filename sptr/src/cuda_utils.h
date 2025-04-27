#ifndef _CUDA_UTILS_H
#define _CUDA_UTILS_H

#include <cmath>
#include <algorithm>
#include <cuda_fp16.h> // 包含 __half 的定义
#ifdef ENABLE_BF16
#include <cuda_bf16.h> // 包含 __nv_bfloat16 的定义
#endif

#define TOTAL_THREADS 1024
#define THREADS_PER_BLOCK 256
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

inline int opt_n_threads(int work_size) {
    const int pow_2 = std::log(static_cast<double>(work_size)) / std::log(2.0);
    return std::max(std::min(1 << pow_2, TOTAL_THREADS), 1);
}

inline dim3 opt_block_config(int x, int y) {
    const int x_threads = opt_n_threads(x);
    const int y_threads = std::max(std::min(opt_n_threads(y), TOTAL_THREADS / x_threads), 1);
    dim3 block_config(x_threads, y_threads, 1);
    return block_config;
}

template <typename T>
__device__ inline T float2type(float value);

template <>
__device__ inline float float2type<float>(float value) {
    return value;
}

template <>
__device__ inline __half float2type<__half>(float value) {
    return __float2half(value);
}

#ifdef ENABLE_BF16
template <>
__device__ inline __nv_bfloat16 float2type<__nv_bfloat16>(float value) {
    return __float2bfloat16(value);
}
#endif

__device__ __forceinline__ __half operator+(const __half &a, const __half &b) {
    return __hadd(a, b);
}

__device__ __forceinline__ __half operator-(const __half &a, const __half &b) {
    return __hsub(a, b);
}

__device__ __forceinline__ __half operator*(const __half &a, const __half &b) {
    return __hmul(a, b);
}

__device__ __forceinline__ __half& operator+=(__half &a, const __half &b) {
    a = __hadd(a, b);
    return a;
}

// #ifdef ENABLE_BF16
// __device__ __forceinline__ __nv_bfloat16 operator+(const __nv_bfloat16 &a, const __nv_bfloat16 &b) {
//     return __hadd(a, b);
// }

// __device__ __forceinline__ __nv_bfloat16 operator-(const __nv_bfloat16 &a, const __nv_bfloat16 &b) {
//     return __hsub(a, b);
// }

// __device__ __forceinline__ __nv_bfloat16 operator*(const __nv_bfloat16 &a, const __nv_bfloat16 &b) {
//     return __hmul(a, b);
// }

// __device__ __forceinline__ __nv_bfloat16& operator+=(__nv_bfloat16 &a, const __nv_bfloat16 &b) {
//     a = __hadd(a, b);
//     return a;
// }
// #endif

#endif
