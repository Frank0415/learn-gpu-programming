#include "common.cuh"
#include <stdio.h>
constexpr int STEP = 4;

__global__ void hello() { printf("GPU Block=%d, Thread=%d!\n", blockIdx.x, threadIdx.x); }

template <int N> int plus_n(int x) { return x + N; }

int main() {
    const int value = 3;
    hello<<<2, 4>>>();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("CPU: kernel completed\n");
    printf("%d\n", plus_n<STEP>(value));
    return 0;
}