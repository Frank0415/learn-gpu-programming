#include "common.cuh"

constexpr int M = 4096;
constexpr int N = 4096;
constexpr int K = 4096;

constexpr int BLOCK = 8;
constexpr int MICROTILE = 4;
constexpr int BLOCK_TILE = BLOCK * MICROTILE;
constexpr int CHUNK = BLOCK;

__host__ __device__ int posToIdx(int x, int y, int dim_x, int dim_y) { return y * dim_x + x; }

__global__ void matmul(float *a, float *b, float *c) {
    int row0 = blockIdx.y * BLOCK_TILE + threadIdx.y * MICROTILE;
    int col0 = blockIdx.x * BLOCK_TILE + threadIdx.x * MICROTILE;

    int chunks = (N + CHUNK - 1) / CHUNK;

    __shared__ float tileA[BLOCK_TILE * CHUNK];
    __shared__ float tileB[CHUNK * BLOCK_TILE];

    float acc[MICROTILE][MICROTILE] = {};

    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    for (int chunk = 0; chunk < chunks; chunk++) {
        for (int elem = tid; elem < BLOCK_TILE * CHUNK; elem += BLOCK * BLOCK) {
            int tile_row = elem / CHUNK, tile_col = elem % CHUNK;
            int global_row = blockIdx.y * BLOCK_TILE + tile_row;
            int global_col = chunk * CHUNK + tile_col;
            tileA[elem] = (global_row < M && global_col < N) ? a[posToIdx(global_col, global_row, N, M)] : 0.0f;
        }
        for (int elem = tid; elem < CHUNK * BLOCK_TILE; elem += BLOCK * BLOCK) {
            int tile_row = elem / BLOCK_TILE, tile_col = elem % BLOCK_TILE;
            int global_row = chunk * CHUNK + tile_row;
            int global_col = blockIdx.x * BLOCK_TILE + tile_col;
            tileB[elem] = (global_row < N && global_col < K) ? b[posToIdx(global_col, global_row, K, N)] : 0.0f;
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < CHUNK; kk++) {
            float fragA[MICROTILE], fragB[MICROTILE];
#pragma unroll
            for (int i = 0; i < MICROTILE; i++)
                fragA[i] = tileA[posToIdx(kk, threadIdx.y * MICROTILE + i, CHUNK, BLOCK_TILE)];
#pragma unroll
            for (int j = 0; j < MICROTILE; j++)
                fragB[j] = tileB[posToIdx(threadIdx.x * MICROTILE + j, kk, BLOCK_TILE, CHUNK)];
#pragma unroll
            for (int i = 0; i < MICROTILE; i++)
#pragma unroll
                for (int j = 0; j < MICROTILE; j++)
                    acc[i][j] += fragA[i] * fragB[j];
        }
        __syncthreads();
    }
    for (int i = 0; i < MICROTILE; i++)
        for (int j = 0; j < MICROTILE; j++)
            if (row0 + i < M && col0 + j < K)
                c[posToIdx(col0 + j, row0 + i, K, M)] = acc[i][j];
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

    dim3 grid((K + BLOCK_TILE - 1) / BLOCK_TILE, (M + BLOCK_TILE - 1) / BLOCK_TILE);
    dim3 block(BLOCK, BLOCK);

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
