#include "common.cuh"
#include <cstddef>
#include <cstdio>
#include <stdio.h>

constexpr int M = 2;
constexpr int N = 11;
constexpr int K = 5;

// c == nullptr: fill a, b and write the reference product into got.
// c != nullptr: compare c against got.
int host_gemm(float *a, float *b, float *got, const float *c, int m, int n, int k);

// Multiply matrix A in size (m,n), B in size (n,k) into C （m,k）
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

int main(int argc, char **argv) {
    (void)argc;
    (void)argv;

    const size_t a_n = (size_t)M * N;
    const size_t b_n = (size_t)N * K;
    const size_t c_n = (size_t)M * K;

    float *a = new float[a_n];
    float *b = new float[b_n];
    float *c = new float[c_n];
    float *got = new float[c_n];

    host_gemm(a, b, got, nullptr, M, N, K);

    float *A = nullptr;
    float *B = nullptr;
    float *C = nullptr;

    CUDA_CHECK(cudaMalloc((void **)&A, a_n * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&B, b_n * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&C, c_n * sizeof(float)));

    dim3 block(16, 8);
    dim3 grid((K + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    CUDA_CHECK(cudaMemcpy(A, a, a_n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B, b, b_n * sizeof(float), cudaMemcpyHostToDevice));
    matmul<<<grid, block>>>(A, B, C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(c, C, c_n * sizeof(float), cudaMemcpyDeviceToHost));
    int rc = host_gemm(a, b, got, c, M, N, K);

    delete[] a;
    delete[] b;
    delete[] c;
    delete[] got;
    CUDA_CHECK(cudaFree(A));
    CUDA_CHECK(cudaFree(B));
    CUDA_CHECK(cudaFree(C));
    return rc;
}

int host_gemm(float *a, float *b, float *got, const float *c, int m, int n, int k) {
    if (c == nullptr) {
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
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < k; ++j) {
                float sum = 0.f;
                for (int t = 0; t < n; ++t) {
                    sum += a[(size_t)i * n + t] * b[(size_t)t * k + j];
                }
                got[(size_t)i * k + j] = sum;
            }
        }
        return 0;
    }

    const size_t count = (size_t)m * k;
    for (size_t idx = 0; idx < count; ++idx) {
        if (c[idx] != got[idx]) {
            int i = (int)(idx / k);
            int j = (int)(idx % k);
            printf("Error at (%d, %d): %g != %g\n", i, j, c[idx], got[idx]);
            return 1;
        }
    }
    printf("PASS\n");
    return 0;
}
