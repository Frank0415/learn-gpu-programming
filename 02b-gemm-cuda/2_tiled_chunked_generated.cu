#include "common.cuh"

constexpr int M = 4096;
constexpr int N = 4096;
constexpr int K = 4096;

constexpr int BM = 32;
constexpr int BN = 32;
constexpr int BK = 8;
constexpr int TM = 4;
constexpr int TN = 4;
constexpr int THREADS = (BM / TM) * (BN / TN);

__host__ __device__ int posToIdx(int x, int y, int dim_x, int dim_y) { return y * dim_x + x; }

__global__ void matmul(float *a, float *b, float *c) {
    int row0 = blockIdx.y * BM + threadIdx.y * TM;
    int col0 = blockIdx.x * BN + threadIdx.x * TN;

    int chunks = (N + BK - 1) / BK;

    __shared__ float tileA[BM * BK];
    __shared__ float tileB[BK * BN];

    float acc[TM][TN] = {};

    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    for (int chunk = 0; chunk < chunks; chunk++) {
        for (int elem = tid; elem < BM * BK; elem += THREADS) {
            int tile_row = elem / BK, tile_col = elem % BK;
            int global_row = blockIdx.y * BM + tile_row;
            int global_col = chunk * BK + tile_col;
            tileA[elem] = (global_row < M && global_col < N) ? a[posToIdx(global_col, global_row, N, M)] : 0.0f;
        }
        for (int elem = tid; elem < BK * BN; elem += THREADS) {
            int tile_row = elem / BN, tile_col = elem % BN;
            int global_row = chunk * BK + tile_row;
            int global_col = blockIdx.x * BN + tile_col;
            tileB[elem] = (global_row < N && global_col < K) ? b[posToIdx(global_col, global_row, K, N)] : 0.0f;
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < BK; kk++) {
            float fragA[TM], fragB[TN];
#pragma unroll
            for (int i = 0; i < TM; i++)
                fragA[i] = tileA[posToIdx(kk, threadIdx.y * TM + i, BK, BM)];
#pragma unroll
            for (int j = 0; j < TN; j++)
                fragB[j] = tileB[posToIdx(threadIdx.x * TN + j, kk, BN, BK)];
#pragma unroll
            for (int i = 0; i < TM; i++)
#pragma unroll
                for (int j = 0; j < TN; j++)
                    acc[i][j] += fragA[i] * fragB[j];
        }
        __syncthreads();
    }
    for (int i = 0; i < TM; i++)
        for (int j = 0; j < TN; j++)
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

    dim3 grid((K + BN - 1) / BN, (M + BM - 1) / BM);
    dim3 block(BN / TN, BM / TM);

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
