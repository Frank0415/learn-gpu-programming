#include "common.cuh"
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

int main() {
  const int total = M * N;
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

  int rc = host_check(a, b, c, M, N);

  free(a);
  free(b);
  free(c);
  CUDA_CHECK(cudaFree(A));
  CUDA_CHECK(cudaFree(B));
  CUDA_CHECK(cudaFree(C));
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
