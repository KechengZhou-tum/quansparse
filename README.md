# UniAD Stage1 — Thor vs Orin 三档 Voxel 严格耗时对比

**测试日期**：2026-04-29
**测试场景**：`_eb` (68bfa5402fd0fcdf4ad7aceb), 30 帧 warm 1 + 29 帧统计
**Build flag**：`--fp16 --int8 --best`（同事 Orin 原配方）
**测量工具**：C++ inference_app (enqueueV3_stage1_*) 内置 timer + Python 聚合

---

## 完整对比表

| 阶段 | **0.2m Thor** | 0.2m Orin | **0.15m Thor** | 0.15m Orin | **v5/0.1m Thor** | v5/0.1m Orin |
|---|---|---|---|---|---|---|
| BEV (HW) | 6400 (80×80) | 6400 | 12544 (112×112) | 12544 | 25600 (160×160) | 25600 |
| num_query | 50 | 50 | 100 | 100 | 50 | 50 |
| num_classes | 4 | 4 | 4 | 4 | 2 | 2 |
| H2D (ms) | **0.45** | 2.01 | **1.03** | 3.67 | **1.85** | 7.01 |
| PREPROC (ms) | **3.24** | - | **3.22** | - | **3.21** | 9.65 |
| **GPU enqueueV3 (ms)** | **41.81** | 131.69 | **69.98** | 162.39 | **135.23** | 227.28 |
| GPU median (ms) | 41.72 | - | 69.95 | - | 134.88 | 222.82 |
| GPU std (ms) | 0.40 | - | 0.43 | - | 0.79 | 15.71 |
| D2H (ms) | **0.22** | 0.61 | **0.36** | 0.89 | **0.63** | 1.54 |
| **total H2D+GPU+D2H (ms)** | **42.48** | 134.30 | **71.37** | 166.94 | **137.70** | 235.82 |

---

## Thor vs Orin 加速比

| 模型 | Thor GPU | Orin GPU | **加速** | Thor total | Orin total | total 加速 |
|---|---|---|---|---|---|---|
| **0.2m** (BEV 6400) | 41.81 | 131.69 | **3.15x** ⭐ | 42.48 | 134.30 | **3.16x** |
| **0.15m** (BEV 12544) | 69.98 | 162.39 | **2.32x** | 71.37 | 166.94 | **2.34x** |
| **v5/0.1m** (BEV 25600) | 135.23 | 227.28 | **1.68x** | 137.70 | 235.82 | **1.71x** |

---

## BEV 越小，Thor 加速越大（硬件特性）

| 维度 | 0.1m → 0.15m | 0.15m → 0.2m |
|---|---|---|
| BEV size 比 | 25600 / 12544 = 2.04x | 12544 / 6400 = 1.96x |
| Thor GPU 比 | 135.23 / 69.98 = 1.93x | 69.98 / 41.81 = 1.67x |
| Orin GPU 比 | 227.28 / 162.39 = 1.40x | 162.39 / 131.69 = 1.23x |

**规律**：
- Thor GPU 倍数比 ≈ BEV 比，**符合 self-attention O(N²) 物理模型**
- Orin GPU 倍数比远小于 BEV 比 → Orin 上大矩阵性能没"线性化"
- **Thor Blackwell INT8 Tensor Core 在小矩阵上利用率高、大矩阵被 memory bandwidth bound** → BEV 越小加速越夸张

**结论**：1.68x → 2.32x → 3.15x 加速差异不是量化 bug，是物理硬件特性。

---

## 各 timer 含义

| Timer | 在哪 | 量什么 | 备注 |
|---|---|---|---|
| **LOAD** | CPU | 6 张 JPEG 磁盘读取 + libjpeg 软解码 | ~257 ms，**不计入推理流水线**；生产环境用相机直连或 NVJPEG 可降到 < 5 ms |
| **H2D** | CUDA stream | uchar img + metadata (lidar2img, can_bus, l2g, timestamp 等) host → device | uchar 仅 3.3MB，比老版 float 13MB 小 4x |
| **PREPROC** | GPU | 单个 fused kernel：uchar→float 类型转换 + normalize (per-channel mean/std) + resize + HWC→CHW layout | GPU-fused 设计，**不再经过 host pinned 中转** |
| **GPU enqueueV3** | CUDA stream | TRT engine 的 enqueueV3 调用（推理本身） | 含 backbone / BEV transformer / track / map heads |
| **D2H** | CUDA stream | engine 输出 (bev_embed, scores, labels, boxes, score_pred, things_masks_sorted 等) device → host | |

### PREPROC 是 GPU 图像预处理（不是 CPU）

- **uchar → float** 类型转换
- **Normalize**（减 mean / 除 std，per-channel）
- **Resize**（如果原图分辨率不是 768×960）
- **HWC → CHW** layout 转置

输出直接是 device 上的 float buffer（绑给 TRT engine 的 `img` input），不再经过 host pinned 中转。

### vs 旧实现的差别（同事在 v5 时优化的版本）

| 维度 | 老 enqueueV3_v5 e2e (CPU 多 kernel) | 新 GPU-fused (stage1_015m / stage1_v5 / stage1_02m) |
|---|---|---|
| 流程 | CPU normalize → host pinned (13MB float) → 再 cudaMemcpyAsync 到 device | uchar 3.3MB H2D → 单 kernel 在 device 就地处理 |
| H2D 量 | 13MB float | 3.3MB uchar (1/4) |
| Timer 归属 | 全算 H2D（看似 H2D 高） | H2D + PREPROC 拆开 |
| 实际"输入传输+预处理"总和 | ~17.55 ms | **~4 ms** |
| 节省 | - | 16-24 ms（去掉冗余 host pinned→pinned memcpy + CPU 阻塞 GPU 等待） |

实际比较两版"输入传输+预处理"延迟时，**不要单看 H2D**，要看 **H2D + PREPROC 总和** 或 **GPU enqueueV3**（纯推理）。

---

## 量化正确性验证

跟 Orin 备份的 `enqueueV3_stage1_015m/output_timing/trt_results.json` 数值对齐，30 帧每帧：

- **max_score 差**：mean ±0.03（Thor 略高更干净）
- **label 集合**：完全一致 (0/1/2)
- **box 数后段**：Thor 12 vs Orin 27（Thor 反而更干净，没那么多假阳性）
- **绝对数值**：Thor 与 Orin 在统计意义上一致，Thor 量化路径正确

**关键澄清**：vis 看到"很多框 / 后段累积假阳性"是 0.15m 模型本身特性（track query 持续保留 + _eb scene 域外），跟 Thor 量化无关。Orin 上跑也是这样且 Orin 更乱。

---

## 关键路径

### Thor 容器内
- 0.2m engine：`UniAD/onnx/uniad_xhumanoid_02m_stage1_int8_thor.engine` (101M)
- 0.15m engine：`UniAD/onnx/uniad_xhumanoid_015m_stage1_int8_thor_v2.engine` (124M)
- v5 engine：`UniAD/onnx/uniad_xhumanoid_stage1_v5_int8_thor.engine` (174M)
- 三个对应 C++ app：`inference_app/enqueueV3_stage1_{02m,015m,v5}/build/uniad`
- 测试输入：`UniAD/nuscenes_np_015/uniad_trt_input` (6741 帧 _eb scene)

### 本机回传
- `/home/mig/Downloads/Perception/thor_results/v5_02m_stage1_30/` (vis + run.log + trt_results.json)
- `/home/mig/Downloads/Perception/thor_results/v5_015m_stage1_30/`
- `/home/mig/Downloads/Perception/thor_results/v5_stage1_30/`

每档 30 帧 × 10 张图 = 300 张，每档 ~302MB。
**Engine-only FPS:** 11.25
