#include "common.cuh"
#include <cstddef>
#include <cstdint>
#include <cstdlib>

constexpr int SIZE = 1024;
constexpr int THREAD = 128;
__global__ void matadd(float *a, float *b, float *out, int n) {
    int pos = blockDim.x * blockIdx.x + threadIdx.x;
    if (pos >= n)
        return;
    out[pos] = a[pos] + b[pos];
}

constexpr int REPS = 100;

void host_fill(float *a, float *b, int n);

int main() {
    int bytes = SIZE * sizeof(float);
    float *a = (float *)host_alloc(bytes);
    float *b = (float *)host_alloc(bytes);
    float *c = (float *)host_alloc(bytes);

    host_fill(a, b, SIZE);

    float *A = nullptr;
    float *B = nullptr;
    float *C = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&A, bytes));
    CUDA_CHECK(cudaMalloc((void **)&B, bytes));
    CUDA_CHECK(cudaMalloc((void **)&C, bytes));
    CUDA_CHECK(cudaMemcpy(A, a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B, b, bytes, cudaMemcpyHostToDevice));
    int block_count = (SIZE + THREAD - 1) / THREAD;
    dim3 grid(block_count);

    cudaEvent_t start, end;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));
    for (int i = 0; i < 5; i++) {
        matadd<<<grid, THREAD>>>(A, B, C, SIZE);
    }
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < REPS; i++) {
        matadd<<<grid, THREAD>>>(A, B, C, SIZE);
    }
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, end));
    float ms_normal = ms / REPS;

    // --- CUDA Graph: record the whole REPS loop once, replay it as one submission ---
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    cudaGraph_t graph;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    for (int i = 0; i < REPS; i++) {
        matadd<<<grid, THREAD, 0, stream>>>(A, B, C, SIZE); // recorded, NOT executed
    }
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));

    cudaGraphExec_t graphExec;
    CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, 0));

    CUDA_CHECK(cudaGraphLaunch(graphExec, stream)); // warm-up replay
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaEventRecord(start, stream));
    CUDA_CHECK(cudaGraphLaunch(graphExec, stream)); // one submission -> 100 kernels
    CUDA_CHECK(cudaEventRecord(end, stream));
    CUDA_CHECK(cudaEventSynchronize(end));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, end));
    float ms_graph = ms / REPS;

    printf("normal loop:  %.4f ms per launch\n", ms_normal);
    printf("graph replay: %.4f ms per kernel  (x%.1f)\n", ms_graph, ms_normal / ms_graph);

    CUDA_CHECK(cudaGraphExecDestroy(graphExec));
    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaStreamDestroy(stream));

    free(a);
    free(b);
    free(c);
    CUDA_CHECK(cudaFree(A));
    CUDA_CHECK(cudaFree(B));
    CUDA_CHECK(cudaFree(C));
    return 0;
}

void host_fill(float *a, float *b, int n) {
    for (int i = 0; i < n; ++i) {
        a[i] = (float)(i % 13);
        b[i] = (float)(i % 7);
    }
}