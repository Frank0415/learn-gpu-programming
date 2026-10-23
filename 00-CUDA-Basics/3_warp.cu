#include "common.cuh"

constexpr int THREADCOUNT = 128;
constexpr int WARP = 4;

void is_correct(int threads, int *sum);
__global__ void tree_sum(int *sum) {
    __shared__ int dataCache[WARP];
    int t = threadIdx.x;
    int x = t + 1;
    for (int offset = 16; offset > 0; offset /= 2)
        x += __shfl_down_sync(0xffffffffu, x, offset);
    if (threadIdx.x % 32 == 0) {
        dataCache[threadIdx.x / 32] = x;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int i = 0; i < 4; i++) {
            *sum += dataCache[i];
        }
    }
}

int main() {
    int *sum;
    int result = 0;
    CUDA_CHECK(cudaMalloc((void **)&sum, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(sum, &result, sizeof(int), cudaMemcpyHostToDevice));
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    tree_sum<<<1, 128>>>(sum);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    CUDA_CHECK(cudaMemcpy(&result, sum, sizeof(int), cudaMemcpyDeviceToHost));
    is_correct(THREADCOUNT, &result);
}

void is_correct(int threads, int *sum) {
    float sumt = 0;
    for (int i = 1; i <= threads; i++) {
        sumt += i;
    }
    if (sumt == *sum) {
        return;
    }
    throw false;
}