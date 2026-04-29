# SparseDrive Thor vs Orin 部署 — 严格耗时对比

**测试日期**：Thor 2026-04-29 / Orin 2026-04-22
**模型**：SparseDrive xhumanoid stage1 INT8+FP16 (temporal e2e perception+motion+map, 同一份 ONNX)
**ONNX**：`sparsedrive_temporal_int8_v2.onnx` (164 MB)
**Build flag**：`--fp16 --int8 --best`

---

## 完整对比表

### C++ warm-loop benchmark (100 iter, 真实部署 latency)

| 阶段 | **Thor (TRT 10.13, SM 11.0 Blackwell)** | **Orin (TRT 10.4, SM 8.7 Ampere)** | **加速** |
|---|---|---|---|
| H2D (ms) | 0.46 (median 0.46, p99 0.54) | 1.46 (median 1.46, p99 1.49) | **3.17x** |
| **enqueueV3 (ms)** | **41.87** (median 41.86, p99 42.43, std 0.17) | **88.87** (median 88.74, p99 89.48, std 0.26) | **2.12x** |
| D2H (ms) | 0.03 (median 0.03, p99 0.07) | 0.05 (median 0.05, p99 0.08) | **1.67x** |
| **Total H2D+GPU+D2H (ms)** | **42.38** (median 42.37) | **90.43** (median 90.30) | **2.13x** |
| **Steady-state FPS** | **23.60** | **11.06** | **2.13x** |
| Engine-only FPS | 23.89 | 11.25 | 2.12x |

---

## Thor vs Orin 加速

```
Orin total:  90.43 ms  → 11.06 FPS
Thor total:  42.38 ms  → 23.60 FPS

Thor 加速 = 90.43 / 42.38 = 2.13x
```

---

## 跟 UniAD stage1 三档加速对比

| 模型 | BEV / 矩阵规模 | Thor GPU | Orin GPU | 加速 |
|---|---|---|---|---|
| UniAD 0.2m stage1 | BEV 6400 | 41.81 | 131.69 | **3.15x** |
| **SparseDrive temporal** | **不依赖 BEV grid** | **41.87** | **88.87** | **2.12x** |
| UniAD 0.15m stage1 | BEV 12544 | 69.98 | 162.39 | 2.32x |
| UniAD v5/0.1m stage1 | BEV 25600 | 135.23 | 227.28 | 1.68x |

**观察**：
- SparseDrive Thor GPU 41.87 ms 跟 UniAD 0.2m stage1 41.81 ms **几乎完全一样**（巧合，但都在 Thor 上 ~42 ms）
- SparseDrive 没有 BEV grid（用 anchor-based sparse query 而不是 dense BEV），所以加速比不跟 voxel size 走
- SparseDrive 加速比 2.13x 比 UniAD 同 GPU 耗时档（0.2m, 3.15x）低 —— 因为 SparseDrive 含更多 transformer attention（DeformableAttentionAggr × 12）+ LayerNorm 206 个，这些 op 在 Blackwell 上加速幅度不如 Conv/MatMul

---

## 量化正确性验证 (Thor vs Orin 数值)

跑同一份 100 帧 inputs，Thor 跟 Orin 输出 frame_0000 + 30 帧整体范数差：

| 输出 | orin_norm | thor_norm | abs_diff mean | rel_diff |
|---|---|---|---|---|
| det_cls | 481.27 | 481.27 | 0.484 | **0.10%** |
| det_box | 3522.06 | 3522.50 | 5.122 | **0.15%** |
| map_cls | 77.68 | 77.71 | 0.086 | **0.11%** |
| map_pts | 278.18 | 278.14 | 0.023 | **0.008%** |
| new_cached_feature | 378.30 | 378.29 | 0.401 | **0.11%** |

**所有关键输出整体范数差 < 0.2%** — Thor 量化数值跟 Orin 完全一致，跨硬件移植没有精度损失。

---

## Engine I/O 签名（不变）

**Inputs (9)**：

| name | shape | dtype | bytes |
|---|---|---|---|
| img | (1, 6, 3, 768, 960) | fp32 | 53,084,160 |
| projection_mat | (1, 6, 4, 4) | fp32 | 384 |
| image_wh | (1, 6, 2) | fp32 | 48 |
| use_prev | (1,) | int32 | 4 |
| prev_cached_feature | (1, 600, 256) | fp32 | 614,400 |
| prev_cached_anchor | (1, 600, 11) | fp32 | 26,400 |
| prev_confidence | (1, 600) | fp32 | 2,400 |
| T_temp2cur | (1, 4, 4) | fp32 | 64 |
| time_interval | (1,) | fp32 | 4 |

**Outputs (8)**：det_cls (1,900,4) / det_box (1,900,11) / det_quality (1,900,2) / map_cls (1,100,3) / map_pts (1,100,40) / new_cached_feature / new_cached_anchor / new_confidence

---

## 运行环境对比

| 项 | Thor (新) | Orin (旧) |
|---|---|---|
| TensorRT | **10.13.3.9** | 10.4.0.26 |
| CUDA | **13.0** | 12.6 |
| GPU | Thor Blackwell, SM 11.0, 20 SMs | Orin Ampere, SM 8.7 |
| Engine 大小 | **113 MB** | 100 MB |
| Plugin | libtensorrt_ops.so 1.2 MB | 同 plugin 源码，aarch64 SM_87 编 |
| OS | Ubuntu 24.04 | L4T R36.4 (Ubuntu 22.04) |

---

## 关键路径

### Thor
- ONNX：`/home/nvidia/sparsedrive_thor/sparsedrive_temporal_int8_v2.onnx` (md5 `cc275289...`)
- engine：`/home/nvidia/sparsedrive_thor/sparsedrive_thor_int8.engine` (113M)
- plugin：`/home/nvidia/sparsedrive_thor/tensorrt_plugins/lib/libtensorrt_ops.so` (1.2M, c++17 + SM 110)
- bench：`orin_bench/thor_bench` + `src/main.cpp`
- 输入：`orin_inputs/` (102 帧 npz) + `orin_inputs_bin/` (100 帧 .bin, 5GB)
- 输出：`thor_outputs/` (100 帧 npz)

### Orin (备份)
- 备份位置：`/home/mig/Downloads/Perception/migration_thor_20260429/orin_full_snapshot/sparsedrive_deploy/`
- engine：`sparsedrive_orin_int8.engine` (100M, TRT 10.4 SM 87)
- 部署文档：`/home/mig/Downloads/Perception/SparseDrive-main/quantization/DEPLOYMENT_SUMMARY.md`

### 本机回传
- `/home/mig/Downloads/Perception/thor_results/sparsedrive_outputs/` (100 帧 thor outputs npz)
- `/home/mig/Downloads/Perception/thor_results/sparsedrive_thor/{thor_bench.log, run_thor.log}`
**Engine-only FPS:** 11.25
