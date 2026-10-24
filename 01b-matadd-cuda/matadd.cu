#include "common.cuh"
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#ifdef __clang__
#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_runtime_wrapper.h>
#endif

constexpr int M = 4096;
constexpr int N = 4096;

constexpr int BlockSize = 256;

void host_fill(float *a, float *b, int n);
int host_check(const float *a, const float *b, const float *got, int m, int n);

__global__ void matadd(float *a, float *b, float *out, int n) {
    int pos = blockDim.x * blockIdx.x + threadIdx.x;
    if (pos >= n)
        return;
    out[pos] = a[pos] + b[pos];
}

__global__ void matadd_vec_float4(float *a, float *b, float *out, int n) {
    int pos = blockDim.x * blockIdx.x + threadIdx.x;
    int base = pos * 4;
    if (base + 3 < n) {
        float4 x = reinterpret_cast<const float4 *>(a)[pos];
        float4 y = reinterpret_cast<const float4 *>(b)[pos];
        reinterpret_cast<float4 *>(out)[pos] =
            make_float4(x.x + y.x, x.y + y.y, x.z + y.z, x.w + y.w);
    } else {
        for (int i = base; i < n; ++i) // tail: at most 3 leftover elements
            out[i] = a[i] + b[i];
    }
}

using add_kernel = void (*)(float *, float *, float *, int);

int main(int argc, char **argv) {
    const int total = argc > 1 ? atoi(argv[1]) : M * N;
    const size_t bytes = (size_t)total * sizeof(float);

    float *a = (float *)host_alloc(bytes);
    float *b = (float *)host_alloc(bytes);
    float *c = (float *)host_alloc(bytes);

    host_fill(a, b, total);

    float *A = nullptr;
    float *B = nullptr;
    float *C = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&A, bytes));
    CUDA_CHECK(cudaMalloc((void **)&B, bytes));
    CUDA_CHECK(cudaMalloc((void **)&C, bytes));

    CUDA_CHECK(cudaMemcpy(A, a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B, b, bytes, cudaMemcpyHostToDevice));

    dim3 grid((total + BlockSize - 1) / BlockSize);
    matadd<<<grid, BlockSize>>>(A, B, C, total);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(c, C, bytes, cudaMemcpyDeviceToHost));

    int rc = host_check(a, b, c, 1, total);

    float *d = (float *)host_alloc(bytes);
    float *D = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&D, bytes));
    dim3 grid_vec((total + (BlockSize * 4) - 1) / (BlockSize * 4));

    // 16B alignment contract: cudaMalloc guarantees it; fall back to scalar if unmet
    bool aligned = reinterpret_cast<uintptr_t>(A) % alignof(float4) == 0 &&
                   reinterpret_cast<uintptr_t>(B) % alignof(float4) == 0 &&
                   reinterpret_cast<uintptr_t>(D) % alignof(float4) == 0;
    if (aligned) {
        matadd_vec_float4<<<grid_vec, BlockSize>>>(A, B, D, total);
    } else {
        matadd<<<grid, BlockSize>>>(A, B, D, total);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(d, D, bytes, cudaMemcpyDeviceToHost));
    rc |= host_check(a, b, d, 1, total);

    const int reps = 50;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    auto time_kernel = [&](add_kernel k, dim3 g) {
        for (int r = 0; r < 5; ++r)
            k<<<g, BlockSize>>>(A, B, D, total);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(start));
        for (int r = 0; r < reps; ++r)
            k<<<g, BlockSize>>>(A, B, D, total);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        return ms / reps;
    };
    float ms_scalar = time_kernel(matadd, grid);
    float ms_vec = time_kernel(matadd_vec_float4, grid_vec);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    double gb = 3.0 * total * sizeof(float);
    printf("matadd:            %.4f ms  (%.1f GB/s)\n", ms_scalar, gb / (ms_scalar * 1e6));
    printf("matadd_vec_float4: %.4f ms  (%.1f GB/s)\n", ms_vec, gb / (ms_vec * 1e6));

    free(a);
    free(b);
    free(c);
    free(d);
    CUDA_CHECK(cudaFree(A));
    CUDA_CHECK(cudaFree(B));
    CUDA_CHECK(cudaFree(C));
    CUDA_CHECK(cudaFree(D));

    return rc;
}

void host_fill(float *a, float *b, int n) {
    for (int i = 0; i < n; ++i) {
        a[i] = (float)(i % 13);
        b[i] = (float)(i % 7);
    }
}

// CPU reference: recompute a[i] + b[i] and compare element by element.
int host_check(const float *a, const float *b, const float *got, int m, int n) {
    const int total = m * n;
    for (int i = 0; i < total; ++i) {
        float ref = a[i] + b[i];
        if (got[i] != ref) {
            printf("Error at (%d, %d): %g != %g\n", i / n, i % n, got[i], ref);
            return 1;
        }
    }
    printf("PASS\n");
    return 0;
}
