#include "common.cuh"
#include <cstdio>

constexpr int M = 4096;
constexpr int N = 4096;
constexpr int K = 4096;

constexpr int SIZE = 16;

__host__ __device__ int pos_to_idx(int x, int y, int dim_x, int dim_y) { return y * dim_x + x; }

__global__ void matmul(float *a, float *b, float *c) {
    int id_y = blockIdx.y * blockDim.y + threadIdx.y;
    int id_x = blockIdx.x * blockDim.x + threadIdx.x;
    if (id_x >= K || id_y >= M) {
        return;
    }

    int num_x = K;
    int num_y = M;
    int id = pos_to_idx(id_x, id_y, num_x, num_y);

    float inc = 0;
    for (int i = 0; i < N; i++) {
        inc += a[pos_to_idx(i, id_y, N, M)] * b[pos_to_idx(id_x, i, K, N)];
    }

    c[id] = inc;
}

int main() {
    int a_size = sizeof(float) * M * N;
    int b_size = sizeof(float) * N * K;
    int c_size = sizeof(float) * M * K;

    float *a_host = new float[M * N];
    float *b_host = new float[N * K];
    float *c_host = new float[M * K];
    for (int i = 0; i < M * N; ++i)
        a_host[i] = (float)(i % 13);
    for (int i = 0; i < N * K; ++i)
        b_host[i] = (float)(i % 7);

    float *a;
    float *b;
    float *c;

    cudaMalloc((void **)&a, a_size);
    cudaMalloc((void **)&b, b_size);
    cudaMalloc((void **)&c, c_size);
    cudaMemcpy(a, a_host, a_size, cudaMemcpyHostToDevice);
    cudaMemcpy(b, b_host, b_size, cudaMemcpyHostToDevice);

    dim3 grid((K + SIZE - 1) / SIZE, (M + SIZE - 1) / SIZE); // x->col(K), y->row(M)
    dim3 block(SIZE, SIZE);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaEvent_t start, end;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));

    cudaGraph_t graph;
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    for (int i = 0; i < 10; i++) {
        matmul<<<grid, block, 0, stream>>>(a, b, c);
    }
    cudaStreamEndCapture(stream, &graph);

    cudaGraphExec_t graphExec;

    cudaGraphInstantiate(&graphExec, graph, 0);
    cudaGraphLaunch(graphExec, stream);
    cudaStreamSynchronize(stream);
    cudaEventRecord(start, stream);
    cudaGraphLaunch(graphExec, stream);
    cudaEventRecord(end, stream);
    cudaEventSynchronize(end);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, end);
    printf("Time: %.2f ms\n", ms / 10);
    printf("Size: M=%d N=%d K=%d  FLOPs=%.3e  %.2f TFLOP/s\n", M, N, K, 2.0 * M * N * K,
           2.0 * M * N * K / (ms / 10 * 1e-3) / 1e12);

    cudaMemcpy(c_host, c, c_size, cudaMemcpyDeviceToHost);

    delete[] a_host;
    delete[] b_host;
    delete[] c_host;
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
}
