# 02a-matmul-tilelang

结论：这是 performance-only（仅性能）扫描，不是 correctness（正确性）测试。
本次没有运行 correctness suite（正确性测试套件）。两轮共测试 34 个配置；34 个性能结果的状态都是 `ok`，但正式 Nsight Compute（NVIDIA GPU 性能分析器）报告只有 12 份。

测试环境和范围：

- 测量日期：2026-08-28。
- GPU：RTX 5070 Ti Laptop / SM120。
- 软件：Torch 2.13.0+cu130、TileLang 0.1.13。
- 工作负载：`4096³`、FP16（16 位浮点数）、`2MNK` FLOPs（浮点运算次数）。

两轮的 cuBLAS（CUDA 的矩阵乘法库）基线来自各自的 `results.json`（原始结果文件）：

| 扫描 | 配置数 | cuBLAS 基线 | TFLOP/s（每秒万亿次浮点运算） | 正式 Nsight 状态 |
|:--|--:|--:|--:|:--|
| 初始扫描 | 12 | 2.9619 ms | 46.4030 | 7 个 `ok`，5 个 `not_run` |
| 扩展扫描 | 22 | 3.0300 ms | 45.3596 | 5 个 `ok`，17 个 `not_run` |

第二轮候选配置中的观测结果如下。它们不是所有形状或所有 GPU 上的全局最优：

| Kernel（内核） | 观测参数 | Duration（耗时） | TFLOP/s | 相对 cuBLAS |
|:--|:--|--:|--:|--:|
| cuBLAS (`torch.matmul`) | 第二轮基线 | 3.0300 ms | 45.3596 | 100.00% |
| `matmul_2` | BM128 BN128 BK64，256 threads（线程） | 2.8711 ms | 47.8692 | 105.53% |
| `matmul_3` | BM128 BN128 BK32，256 threads（线程），2 stages（流水级） | 2.7126 ms | 50.6661 | 111.70% |

完整的性能表和日志：

- [初始 12 个配置](../Trash/profiles/02a-matmul/20260828-024614/results.md)
- [扩展 22 个配置](../Trash/profiles/02a-matmul/20260828-033128/results.md)

上表中高于 cuBLAS 的数值只表示这次性能测量的结果。它们不是 correctness（正确性）结论，因为本次扫描没有验证数值输出。

参数保存在 [`performance_configs.json`](performance_configs.json)；扫描程序不会改写 kernel（内核）的默认参数。

运行：

```bash
make sweep
make profile
```

`make profile` 使用 Nsight Compute 的 `--set full`。由于本机设置了 `RmProfilingAdminOnly=1`，命令会打开图形化 sudo 提示。

正式 Nsight 报告只保留已核对 `Kernel Name`（内核名称）为 `matmul_2_kernel` 或 `matmul_3_kernel` 的结果：初始扫描保留 7 份，扩展扫描保留 5 份。其余 5 份和 17 份显示为 `not_run`；未核验的 profiling（性能分析）产物隔离在各轮的 `nsight/invalid-random-kernel/` 目录中，不作为正式报告。
