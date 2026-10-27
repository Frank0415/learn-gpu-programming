#include "common.cuh"
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#ifdef __clang__
#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_runtime_wrapper.h>
#endif

constexpr int LIMIT = 1024 * 1024;

constexpr int M = 16384;
constexpr int N = 16384;

constexpr int BlockSize = 256;

__global__ void init(float *a, float *b, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n)
        return;
    a[i] = (float)(i % 13);
    b[i] = (float)(i % 7);
}

__global__ void matadd(float *a, float *b, float *out, int n) {
    int pos = blockDim.x * blockIdx.x + threadIdx.x;
    if (pos >= n)
        return;
    out[pos] = a[pos] + b[pos];
}

// Instead of launching threads proportional to size of matrix,
// Launch a number of threads and use grid-stride
__global__ void matadd_vec_float4(float *a, float *b, float *out, int n) {
    int pos = blockDim.x * blockIdx.x + threadIdx.x;
    int base = pos * 4;
    int stride = blockDim.x * gridDim.x * 4;
    for (int i = base; i < n; i += stride) {
        if (i < n - 3) { // can be applied to float4
            // DO NOT use a[i], as i is multiplied by 4.
            float4 x = *reinterpret_cast<const float4 *>(a + i);
            float4 y = *reinterpret_cast<const float4 *>(b + i);
            *reinterpret_cast<float4 *>(out + i) =
                make_float4(x.x + y.x, x.y + y.y, x.z + y.z, x.w + y.w);
        } else {
            for (int j = i; j < n; j++) {
                out[j] = a[j] + b[j];
            }
        }
    }
}

// device-side check: scalar matadd output C is the reference; compare D element-wise
__global__ void compare(const float *ref, const float *got, int n, int *bad) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n && ref[i] != got[i])
        atomicExch(bad, 1);
}

using add_kernel = void (*)(float *, float *, float *, int);

int main(int argc, char **argv) {
    const int total = argc > 1 ? atoi(argv[1]) : M * N;
    const size_t bytes = (size_t)total * sizeof(float);

    float *A = nullptr;
    float *B = nullptr;
    float *C = nullptr;
    float *D = nullptr;
    int *d_bad = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&A, bytes));
    CUDA_CHECK(cudaMalloc((void **)&B, bytes));
    CUDA_CHECK(cudaMalloc((void **)&C, bytes));
    CUDA_CHECK(cudaMalloc((void **)&D, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_bad, sizeof(int)));

    dim3 grid((total + BlockSize - 1) / BlockSize);
    dim3 grid_vec;
    if (total > LIMIT) {
        grid_vec = dim3(LIMIT / (BlockSize * 4));
    } else {
        grid_vec = dim3((total + (BlockSize * 4) - 1) / (BlockSize * 4));
    }
    init<<<grid, BlockSize>>>(A, B, total);
    matadd<<<grid, BlockSize>>>(A, B, C, total); // reference implementation

    bool aligned = reinterpret_cast<uintptr_t>(A) % alignof(float4) == 0 &&
                   reinterpret_cast<uintptr_t>(B) % alignof(float4) == 0 &&
                   reinterpret_cast<uintptr_t>(D) % alignof(float4) == 0;
    if (aligned) {
        matadd_vec_float4<<<grid_vec, BlockSize>>>(A, B, D, total);
    } else {
        matadd<<<grid, BlockSize>>>(A, B, D, total);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemset(d_bad, 0, sizeof(int)));
    compare<<<grid, BlockSize>>>(C, D, total, d_bad);
    int bad = 0;
    CUDA_CHECK(cudaMemcpy(&bad, d_bad, sizeof(int), cudaMemcpyDeviceToHost));
    int rc = bad ? 1 : 0;
    printf(bad ? "FAIL\n" : "PASS\n");

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

    CUDA_CHECK(cudaFree(A));
    CUDA_CHECK(cudaFree(B));
    CUDA_CHECK(cudaFree(C));
    CUDA_CHECK(cudaFree(D));
    CUDA_CHECK(cudaFree(d_bad));

    return rc;
}
