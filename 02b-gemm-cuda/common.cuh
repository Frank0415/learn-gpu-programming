#pragma once
#include <cuda_runtime.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

inline void cuda_check(cudaError_t status, const char *call, const char *file, int line) {
    if (status != cudaSuccess) {
        fprintf(stderr, "%s:%d: %s: %s\n", file, line, call, cudaGetErrorString(status));
        exit(EXIT_FAILURE);
    }
}
#define CUDA_CHECK(call) cuda_check((call), #call, __FILE__, __LINE__)

inline int positive_arg(const char *text) {
    char *end = NULL;
    errno = 0;
    long value = strtol(text, &end, 10);
    if (errno || end == text || *end || value < 1 || value > INT_MAX) {
        fprintf(stderr, "Expected a positive integer <= INT_MAX, got: %s\n", text);
        exit(EXIT_FAILURE);
    }
    return (int)value;
}

inline void *host_alloc(size_t bytes) {
    void *p = malloc(bytes);
    if (!p) {
        fprintf(stderr, "Host allocation failed: %zu bytes\n", bytes);
        exit(EXIT_FAILURE);
    }
    return p;
}
