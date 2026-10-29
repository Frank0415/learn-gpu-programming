#include "common.cuh"

constexpr int M = 4096;
constexpr int N = 4096;
constexpr int K = 4096;

constexpr int SIZE = 16;

__host__ __device__ int posToIdx(int x, int y, int dim_x, int dim_y) { return y * dim_x + x; }

// Every time collab to take a chunk of SIZE * SIZE,
// and total of SIZE * N from Matrix a and N * SIZE from Matrix b,
// and split in N's direction

__global__ void matmul(float *a, float *b, float *c) {
    int id_y = blockIdx.y * blockDim.y + threadIdx.y;
    int id_x = blockIdx.x * blockDim.x + threadIdx.x;

    int iters = (N + SIZE - 1) / SIZE;

    __shared__ float tileA[SIZE * SIZE];
    __shared__ float tileB[SIZE * SIZE];

    int num_x = K;
    int num_y = M;
    int idA, idB;
    int idC = posToIdx(id_x, id_y, num_x, num_y);

    int idShared = posToIdx(threadIdx.x, threadIdx.y, SIZE, SIZE);
    float inc = 0;

    for (int j = 0; j < iters; j++) {
        idA = posToIdx(threadIdx.x + j * SIZE, id_y, N, M);
        idB = posToIdx(id_x, j * SIZE + threadIdx.y, K, N);
        tileA[idShared] = a[idA];
        tileB[idShared] = b[idB];
        __syncthreads();
        for (int i = 0; i < SIZE; i++) {
            float ax = tileA[threadIdx.y * blockDim.x + i];
            float bx = tileB[i * blockDim.y + threadIdx.x];

            inc += ax * bx;
        }
        __syncthreads();
    }
    if (id_x >= K || id_y >= M) {
        return;
    }
    c[idC] = inc;
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
