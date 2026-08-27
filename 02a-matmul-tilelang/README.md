# 02a-matmul-tilelang

Platform: NVIDIA 5070Ti Laptop, CUDA 13.3, M = N = K = 4096, fp32

| Kernel Name | Flops | vs CuBLAS |
|:-------|:------|:------|
|CuBLAS|9.998 ms| 100.00%|
|matmul_1: Native matmul | 127.710 ms| 7.829%|