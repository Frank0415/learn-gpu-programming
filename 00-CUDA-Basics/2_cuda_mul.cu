#include "common.cuh"
#include <cstddef>
#include <cstdio>
#include <stdio.h>

#define DEBUG 0

// #ifdef DEBUG

// constexpr int M = 4;
// constexpr int N = 8;
// constexpr int K = 4;

// constexpr int TILE = 2;

// #else

constexpr int M = 4096;
constexpr int N = 4096;
constexpr int K = 4096;

constexpr int TILE = 16;

// #endif

void host_fill(float *a, float *b, int m, int n, int k);
int host_compare(const float *ref, const float *got, int m, int k);

using matmul_kernel = void (*)(float *, float *, float *, int, int, int);
float test_time_matmul(matmul_kernel kernel, float *A, float *B, float *C, int m, int n, int k,
                       int iters);

// Multiply matrix A in size (m,n), B in size (n,k) into C（m,k）
__global__ void matmul(float *A, float *B, float *C, int m, int n, int k) {
    int pos_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pos_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (pos_x >= k || pos_y >= m)
        return;

    int c_pos = pos_y * k + pos_x;

    float acc = 0.0f;

    for (int i = 0; i < n; i++) {
        float a = A[pos_y * n + i];
        float b = B[i * k + pos_x];

        acc += a * b;
    }

    C[c_pos] = acc;
}

// Multiply matrix A in size (m,n), B in size (n,k) into C（m,k）
// For a tile size of (blockDim.x, blockDim.y), we load the related memory to __shared__ A_cache and
// B_cache
__global__ void matmul_shared_block(float *A, float *B, float *C, int m, int n, int k) {
    constexpr int size = TILE * TILE;
    __shared__ float A_cache[size];
    __shared__ float B_cache[size];

    int pos_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pos_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (pos_x >= k || pos_y >= m)
        return;

    int blocks = (n + blockDim.x - 1) / TILE;

    float acc = 0.0f;

    int A_pos = blockDim.y * blockIdx.y * n;
    int B_pos = blockIdx.x * blockDim.x;
    int cache_pos = threadIdx.y * blockDim.x + threadIdx.x;
    int glob_pos_A = threadIdx.y * n + threadIdx.x;
    int glob_pos_B = threadIdx.y * k + threadIdx.x;
    for (int i = 0; i < blocks; i++) {

        // bring the data into __shared__
        A_cache[cache_pos] = A[A_pos + glob_pos_A];
        B_cache[cache_pos] = B[B_pos + glob_pos_B];

#if DEBUG
        if (blockIdx.x == 0 && blockIdx.y == 1) {
            printf("Index: %d\nMoved global position A(%d,%d) into A_cache(%d,%d), B(%d,%d) into "
                   "B_cache(%d,%d)\n",
                   i, (A_pos + glob_pos_A) % n, (A_pos + glob_pos_A) / n, threadIdx.y, threadIdx.x,
                   (B_pos + glob_pos_B) % k, (B_pos + glob_pos_B) / k, threadIdx.y, threadIdx.x);
        }
#endif

        A_pos += blockDim.x;
        B_pos += k * blockDim.y;
        __syncthreads();
        for (int i = 0; i < TILE; i++) {
            float a = A_cache[threadIdx.y * blockDim.x + i];
            float b = B_cache[i * blockDim.y + threadIdx.x];

            acc += a * b;
        }
        __syncthreads();
    }

    int c_pos = pos_y * k + pos_x;
    C[c_pos] = acc;
}

int main(int argc, char **argv) {
    (void)argc;
    (void)argv;

    const size_t a_n = (size_t)M * N;
    const size_t b_n = (size_t)N * K;
    const size_t c_n = (size_t)M * K;

    float *a = new float[a_n];
    float *b = new float[b_n];
    float *c = new float[c_n];
    float *ref = new float[c_n];

    host_fill(a, b, M, N, K);

    float *A = nullptr;
    float *B = nullptr;
    float *C = nullptr;

    CUDA_CHECK(cudaMalloc((void **)&A, a_n * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&B, b_n * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&C, c_n * sizeof(float)));

    dim3 block(TILE, TILE);
    dim3 grid((K + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    CUDA_CHECK(cudaMemcpy(A, a, a_n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B, b, b_n * sizeof(float), cudaMemcpyHostToDevice));

    // Naive kernel produces the reference output.
    matmul<<<grid, block>>>(A, B, C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(ref, C, c_n * sizeof(float), cudaMemcpyDeviceToHost));

    // Tiled kernel under test.
    matmul_shared_block<<<grid, block>>>(A, B, C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(c, C, c_n * sizeof(float), cudaMemcpyDeviceToHost));
    int rc = host_compare(ref, c, M, K);

    const int iters = 10;
    float ms = test_time_matmul(matmul, A, B, C, M, N, K, iters);
    float ms_shared = test_time_matmul(matmul_shared_block, A, B, C, M, N, K, iters);
    printf("matmul:               %.3f ms  (%.2f GFLOP/s)\n", ms, 2.0 * M * N * K / (ms * 1e6));
    printf("matmul_shared_block:  %.3f ms  (%.2f GFLOP/s)\n", ms_shared,
           2.0 * M * N * K / (ms_shared * 1e6));

    delete[] a;
    delete[] b;
    delete[] c;
    delete[] ref;
    CUDA_CHECK(cudaFree(A));
    CUDA_CHECK(cudaFree(B));
    CUDA_CHECK(cudaFree(C));
    return rc;
}

// Returns the average kernel time in milliseconds over `iters` launches.
// `kernel` is a host-side pointer to a __global__ function.
float test_time_matmul(matmul_kernel kernel, float *A, float *B, float *C, int m, int n, int k,
                       int iters) {
    dim3 block(TILE, TILE);
    dim3 grid((k + block.x - 1) / block.x, (m + block.y - 1) / block.y);

    kernel<<<grid, block>>>(A, B, C, m, n, k); // warmup
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; i++) {
        kernel<<<grid, block>>>(A, B, C, m, n, k);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / iters;
}

void host_fill(float *a, float *b, int m, int n, int k) {
    for (int i = 0; i < m; ++i) {
        for (int t = 0; t < n; ++t) {
            a[(size_t)i * n + t] = (float)((i + t) % 13);
        }
    }
    for (int t = 0; t < n; ++t) {
        for (int j = 0; j < k; ++j) {
            b[(size_t)t * k + j] = (float)((t + 2 * j) % 7);
        }
    }
}

int host_compare(const float *ref, const float *got, int m, int k) {
    const size_t count = (size_t)m * k;
    for (size_t idx = 0; idx < count; ++idx) {
        if (got[idx] != ref[idx]) {
            int i = (int)(idx / k);
            int j = (int)(idx % k);
            printf("Error at (%d, %d): %g != %g\n", i, j, got[idx], ref[idx]);
            return 1;
        }
    }
    printf("PASS\n");
    return 0;
}
