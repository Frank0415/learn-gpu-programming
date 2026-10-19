#include "common.cuh"
#include <chrono>
#include <iostream>
#include <stdio.h>

__global__ void multiply_cuda(float *content, int size) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size) {
        content[pos] *= 2.0f;
    }
}

int main(int argc, char **argv) {
    int size = argc > 1 ? positive_arg(argv[1]) : 1003;
    float *content = new float[size];
    float *check = new float[size];
    float *check_buffer = new float[size];

    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < size; i++) {
        content[i] = i;
        check[i] = i * 2;
    }
    auto end = std::chrono::steady_clock::now();
    double us = std::chrono::duration<double, std::micro>(end - start).count();
    std::cout << "Host Time: " << us << " us" << std::endl;

    float *data = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&data, size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy((void *)data, (const void *)content, size * sizeof(float),
                          cudaMemcpyHostToDevice));

    int blocksize[7] = {1024, 256, 128, 64, 32, 8, 1};
    multiply_cuda<<<(size + 255) / 256, 256>>>(data, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    for (auto sz : blocksize) {
        CUDA_CHECK(cudaMemcpy(data, content, size * sizeof(float), cudaMemcpyHostToDevice));
        auto start = std::chrono::steady_clock::now();

        multiply_cuda<<<(size + sz - 1) / sz, sz>>>(data, size);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        auto end = std::chrono::steady_clock::now();
        double us = std::chrono::duration<double, std::micro>(end - start).count();
        std::cout << "block " << sz << " threads Time: " << us << " us" << std::endl;

        cudaMemcpy(check_buffer, data, size * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < size; i++) {
            if (check_buffer[i] != check[i]) {
                std::cout << "Error at index " << i << " : " << check_buffer[i]
                          << " != " << check[i] << std::endl;
                return 1;
            }
        }
    }

    CUDA_CHECK(
        cudaMemcpy((void *)content, (void *)data, size * sizeof(float), cudaMemcpyDeviceToHost));
    delete[] content;
    delete[] check;
    delete[] check_buffer;
    CUDA_CHECK(cudaFree((void *)data));
    return 0;
}
