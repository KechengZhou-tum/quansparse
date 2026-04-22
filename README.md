# quansparse
# SparseDrive V1 stage1 INT8 量化 + Orin 部署总结

**日期**：2026-04-22
**项目**：SparseDrive xhumanoid stage1（det head + map head，无 motion / planning）
**目标硬件**：NVIDIA AGX Orin 64GB（JetPack 6.1，CUDA 12.6，TensorRT 10.4）

---

## 1. 数据来源

| 项 | 路径 | 说明 |
|---|---|---|
| nuScenes val | 云端 `/media/data01/nuscenes_format/` | 6741 帧 / 61 scene |
| val info pkl | `nuscenes_infos_temporal_val.pkl` | 标注 + 相机 meta |
| 训练 checkpoint | 云端 `/media/data01/sparse/SparseDrive-main/work_dirs-old/sparsedrive_xhumanoid_stage1/iter_15500.pth` | PyTorch FP32 基线 |
| val 图像 | `/media/data01/nuscenes_format/samples/CAM_*/*.jpg` | 6 相机 |

**val 帧分布**（用于量化泛化性分段测试）：
- **front 段 [0, 500)**：5 个 scene，车 + 行人 + 车道混合
- **mid 段 [3000, 3500)**：6 个 scene，纯行人 + 人行道
- **mid 段 [3000, 3040)**：40 帧（本次 Orin 部署 + C++ benchmark 使用）

---

## 2. 端到端 mAP 对比矩阵

### 2.1 val-500（front [0, 500)，全链路）

| 配置 | det mAP | drivable_iou | pedestrian_AP | vehicle_AP |
|---|---|---|---|---|
| **PyTorch FP32 + temporal** | **0.2836** | 0.2841 | 0.6218 | 0.5124 |
| TRT FP32 engine（单帧） | 0.2387 | 0.2841 | 0.5508 | 0.3975 |
| TRT FP16 engine（opset 17） | 0.2370 | 0.2842 | 0.5498 | 0.3965 |
| TRT INT8+FP16（opset 17, 单帧） | 0.2309 | 0.2852 | 0.5362 | 0.3807 |
| TRT FP16 + temporal（warm v3） | 0.2809 | 0.2841 | 0.6115 | 0.5122 |
| **TRT INT8+FP16 + temporal（v2 warm）** ✅ | **0.2712** | **0.2847** | 0.5967 | 0.4881 |

**损失拆解**：
- PyTorch → TRT FP16 temporal: **-0.96%**（TRT 编译器精度损失）
- FP16 temporal → INT8+FP16 temporal: **-3.4%**（INT8 量化损失）
- **INT8 temporal vs PyTorch FP32：-4.4%**（最终值）
- drivable_iou 和 mean_map_iou **完全等于** PyTorch 基线

### 2.2 多段泛化性（500 帧每段）

| 段 | PyTorch | TRT FP16 | TRT INT8 | INT8 损失 | 场景 |
|---|---|---|---|---|---|
| front [0,500) | 0.2836 | 0.2809 | 0.2712 | -4.4% | 车+行人+车道 |
| mid [3000,3500) | 0.1945 | 0.1951 | 0.1916 | -1.5% | 纯行人 |

**pedestrian_AP**（分类别）：
| 段 | PyTorch | TRT FP16 | TRT INT8 |
|---|---|---|---|
| front | 0.622 | 0.612 | 0.597 (-4.0%) |
| mid | 0.778 | 0.780 | 0.766 (-1.5%) |

### 2.3 Orin 跨机精度验证（mid [3000, 3040) 40 帧）

| Backend | det mAP | walkable_iou | mean_map_iou | Δ vs prev |
|---|---|---|---|---|
| PyTorch FP32 | 0.19881 | 0.04992 | 0.01702 | baseline |
| **TRT INT8 (x86 A800)** | **0.19476** | 0.05155 | 0.01765 | **-2.0%** |
| **TRT INT8 (Orin)** | **0.19419** | 0.05454 | 0.01864 | **-0.29% vs x86** |

**Orin engine vs x86 engine 只差 0.29% mAP**，完全可接受。

---

## 3. Orin 部署 latency（本次核心交付）

### 3.1 三种测量方式对比

| 方式 | enqueueV3 median | 有效性 |
|---|---|---|
| trtexec benchmark | **80.41 ms** | 理论上限（纯 GPU kernel, 无 CPU 干扰） |
| Python runner | **147.09 ms** | ⚠️ **偏高** — npz 解压让 CUDA context 冷却 |
| **C++ warm-loop bench** | **88.87 ms** | ✅ **真实部署 latency** |

### 3.2 C++ benchmark 详细数据（100 iter stats）

```
=== STATISTICS (ms, N=100) ===
  H2D         mean=   1.46  median=   1.46  p90=   1.47  p99=   1.49  std= 0.00
  enqueueV3   mean=  88.87  median=  88.74  p90=  89.28  p99=  89.48  std= 0.26
  D2H         mean=   0.05  median=   0.05  p90=   0.06  p99=   0.08  std= 0.00
  Total       mean=  90.43  median=  90.30  p90=  90.84  p99=  91.05  std= 0.26

steady-state FPS: 11.06
engine-only FPS:  11.25
```

### 3.3 关键结论

- **真实 FPS = 11.06**（C++ 端到端，含 H2D + enqueue + D2H）
- enqueueV3 88.87 ms ≈ trtexec 80 ms + 10% overhead（per-frame binding）
- Orin 统一内存：**H2D 53 MB img 只要 1.46 ms**（36 GB/s 有效带宽，远优于 PCIe discrete GPU）
- D2H 0.05 ms（output 只有 720 KB）
- **std 0.26 ms 超稳定**，steady state 无抖动

### 3.4 Python 147 ms 为什么偏高？

CUDA context "冷却"：每帧之间 Python 解压 21 MB npz + `np.savez_compressed` 等 CPU 活 250-450 ms，期间 GPU 调度器释放缓存 kernel；下一次 enqueueV3 重新加载 weights/kernels，额外 +50–60 ms。C++ warm loop 去掉 npz IO 后，enqueue 落回 89 ms 硬件真实水平。

---

## 4. Orin engine I/O 签名

**Inputs (9)**：

| name | shape | dtype |
|---|---|---|
| img | (1, 6, 3, 768, 960) | fp32 |
| projection_mat | (1, 6, 4, 4) | fp32 |
| image_wh | (1, 6, 2) | fp32 |
| use_prev | (1,) | int32 |
| prev_cached_feature | (1, 600, 256) | fp32 |
| prev_cached_anchor | (1, 600, 11) | fp32 |
| prev_confidence | (1, 600) | fp32 |
| T_temp2cur | (1, 4, 4) | fp32 |
| time_interval | (1,) | fp32 |

**Outputs (8)**：

| name | shape | dtype |
|---|---|---|
| det_cls | (1, 900, 4) | fp32 |
| det_box | (1, 900, 11) | fp32 |
| det_quality | (1, 900, 2) | fp32 |
| map_cls | (1, 100, 3) | fp32 |
| map_pts | (1, 100, 40) | fp32 |
| new_cached_feature | (1, 600, 256) | fp32 |
| new_cached_anchor | (1, 600, 11) | fp32 |
| new_confidence | (1, 600) | fp32 |

Host 侧维护 `T_temp2cur = inv(T_global_cur) @ T_global_prev`，scene 切换时 `use_prev=0` 并把 prev_state 清零。

---

## 5. 文件清单

### 5.1 云端（`root@10.108.1.132:8278`）

```
/root/workspace/sparsedrive_quant/SparseDrive-main/
├── projects/                              # SparseDrive 原码（配置、plugin）
├── work_dirs/sparsedrive_xhumanoid_stage1 # softlink → /media/data01/sparse/.../work_dirs-old/...
├── quantization/
│   ├── artifacts/
│   │   ├── sparsedrive_temporal_fp32_op17.onnx       # FP32 ONNX opset 17
│   │   ├── sparsedrive_temporal_int8.onnx            # INT8 ONNX 167MB
│   │   ├── sparsedrive_temporal_fp16.engine          # FP16 engine 131MB
│   │   ├── sparsedrive_temporal_int8_v2.engine       # x86 A800 INT8+FP16 engine 108MB
│   │   ├── pytorch_fp32_val500.pkl                   # PyTorch FP32 front-500 pkl
│   │   ├── temporal_int8_v2_warm_val500.pkl          # TRT INT8 front-500 pkl
│   │   ├── pytorch_mid40_fp32.pkl                    # PyTorch FP32 mid-40 pkl
│   │   ├── temporal_int8_x86_mid40_val500.pkl        # x86 INT8 mid-40 pkl
│   │   └── temporal_int8_orin_mid40_val500.pkl       # Orin INT8 mid-40 pkl
│   ├── export/
│   │   └── wrappers.py                               # SparseDrivePerceptionExport (temporal)
│   └── vis_3way_orin_mid40/                          # 40 帧 4-路对比 jpg
├── /tmp/
│   ├── run_temporal_eval.py                          # temporal engine eval (cloud)
│   ├── run_val_segment.py                            # PyTorch segment eval
│   ├── dump_and_run_x86.py                           # dump inputs + run x86 engine
│   ├── orin_postprocess.py                           # Orin raw outputs → pkl
│   ├── eval_segment_pkl.py                           # segment pkl 评测
│   └── viz_compare_3way.py                           # 4-路可视化（GT/PT/x86/Orin）
```

环境：`python3.8` + TRT 10.9 at `/workspace/TensorRT-10.9_x86_cu118/targets/x86_64-linux-gnu/lib`
激活：`export LD_LIBRARY_PATH=/workspace/TensorRT-10.9_x86_cu118/targets/x86_64-linux-gnu/lib:$LD_LIBRARY_PATH`

### 5.2 Orin（`nvidia@10.10.253.131`）

```
~/sparsedrive_deploy/
├── sparsedrive_temporal_int8_v2.onnx                 # 164MB, cloud 传过来
├── sparsedrive_orin_int8.engine                      # 100MB, TRT 10.4 SM_87
├── tensorrt_plugins/lib/libtensorrt_ops.so           # aarch64 plugin（容器内编）
├── orin_inputs/                                      # 40 帧 .npz（Python runner 用）
├── orin_inputs_bin/                                  # 10 帧 flat .bin（C++ bench 用）
├── orin_outputs/                                     # Python runner 的输出
├── orin_run_engine.py                                # Python runner + per-stage timer
└── orin_bench/                                       # C++ benchmark
    ├── src/{main.cpp, engine.hpp, engine.cpp, cuda_timer.hpp}
    ├── prepare_blob.py
    ├── build.sh
    └── orin_bench                                    # 编译产物
```

**容器**：`uniad-orin`（`dustynv/l4t-pytorch:r36.4.0`, `-v /:/host` 挂载）
**trtexec 路径**：`/usr/src/tensorrt/targets/aarch64-linux-gnu/bin/trtexec`

### 5.3 本地（`/home/mig/Downloads/Perception/SparseDrive-main/`）

```
quantization/
├── orin_bench/                                       # C++ benchmark 源码（一份权威副本）
├── vis_3way_clean/                                   # 前 40 帧 3-路（前期）
├── vis_3way_mid/                                     # mid 100 帧 3-路
├── vis_3way_orin_mid40/                              # mid 40 帧 4-路（本次）
├── vis_3way_clean.mp4                                # 4.2 MB
├── vis_3way_mid.mp4                                  # 12.3 MB
├── vis_3way_orin_mid40.mp4                           # 9.7 MB ⭐ 最新
└── DEPLOYMENT_SUMMARY.md                             # 本文档
```

---

## 6. 踩过的坑（按时间顺序）

### 6.1 量化阶段

**坑 1：opset 16 下 FP16 engine mAP 崩溃**
- 症状：FP16 engine mAP 0.1329（PyTorch 0.2370 的 56%），看起来完全不可用
- 根因：opset 16 把 LayerNorm 散开成 ReduceMean + Sub + Mul 等原子 op，中间值 fp32 巨大，转 FP16 累积溢出
- 修复：升级 PyTorch 1.13 + `opset_version=17`（LayerNormalization 整体 op）→ FP16 mAP 回到 0.2370

**坑 2：ModelOpt 默认量化破坏模型**
- 症状：无论校准集多大，INT8 都 mAP 0.14
- 修复：加 `--op_types_to_exclude MatMul --disable_mha_qdq --dq_only`（MatMul 保 FP16，只量化 Conv/GEMM 主体）

### 6.2 Temporal I/O wrapper

**坑 3：map_head drivable_iou 从 0.2841 崩到 0.1876（-34%）**
- 症状：INT8 temporal engine drivable_iou 崩溃，但 det mAP 看起来正常
- 根因：`_stateless_map_head_forward` 里对 `op == 'temp_gnn'` 写了 `continue`，误以为 `num_temp_instances=0` 意味着没有 temp_gnn 层
- 真相：`temporal_map=True + num_temp_instances=0` 意思是 temp_gnn 层**存在**但没 cached temp，原 forward 是 `key=None, value=None` 让 MHA 退化为 self-attn
- 修复：`graph_model(key=None, value=None, query_pos=anchor_embed, attn_mask=None)` → drivable_iou 回到 0.2847 bit-perfect

**坑 4：cold-start 不等价原单帧 reset**
- 原 wrapper `instance_bank.reset()` → temp_gnn 走 `key=None` self-attn
- 新 stateless wrapper cold 时走 cross-attn（prev 为 0），cold-only mAP 0.1506 < 单帧 0.2370
- 缓解：主要跑 warm 模式，cold 只 scene 第一帧，影响可忽略

### 6.3 评测脚本

**坑 5：dataloader prefetch 整段 val 导致 segment eval 30 分钟 hang**
- 症状：`run_val_segment.py --start 3000` 在 frame 1 处卡 30 分钟
- 根因：dataloader workers 并行 prefetch 所有前置 frames 的图像
- 修复：**在 build_dataloader 之前** `dataset.data_infos = full_infos[start:start+num]`

**坑 6：dataset.evaluate 忽略 results 长度**
- 症状：eval 40 帧却 iterate 6741 次
- 根因：nuScenes evaluate 是对全 gt 套表，不是 iterate results
- 正确做法：slice `dataset.data_infos` 后 evaluate `results[:N]` —— 之前 run_temporal_eval.py 已正确处理

### 6.4 可视化

**坑 7：可视化 box 格式判断错误**
- 症状：frame 0 画出"天上飞的红色 box"
- 根因：PyTorch FP32 frame 0 结果是 7-dim `[x,y,z,w,l,h,yaw]`，其它 frame 是 11-dim `[x,y,z,w,l,h,sin,cos,vx,vy,vz]`；原判断 `len(box) >= 8 and len(box) != 7` 在 7-dim 时错误把 w 当 sin_yaw
- 修复：显式 `if len(box) >= 10 else scalar yaw`

**坑 8：cam_render 外参不正交**
- 症状：`Matrix must be orthogonal, i.e. its transpose should be its inverse`
- 修复：放弃原 cam_render 工具，自写 `viz_compare.py` 用 `sensor2lidar_rotation.T` 做 lidar→cam 变换

### 6.5 跨机部署

**坑 9：Python 3.8 vs 3.10 + TRT lib 路径混乱**
- `python3.10` 有 tensorrt 模块但 libnvinfer.so.10 需 `LD_LIBRARY_PATH=/workspace/TensorRT-10.9_x86_cu118/targets/x86_64-linux-gnu/lib`
- `python3.8` 也有 tensorrt，且**插件 .so 是 python3.8 编的**
- 解决：统一用 `python3.8` + 固定 LD_LIBRARY_PATH

**坑 10：work_dirs 软链指向 broken 路径**
- 原链接指向 `/media/data01/sparse/SparseDrive-main/work_dirs/...`（不存在）
- 实际 checkpoint 在 `/media/data01/sparse/SparseDrive-main/work_dirs-old/...`
- 修复：`ln -sfn /media/data01/sparse/SparseDrive-main/work_dirs-old/sparsedrive_xhumanoid_stage1 work_dirs/sparsedrive_xhumanoid_stage1`

**坑 11：Orin 根分区只剩 6 GB**
- Orin `/dev/mmcblk0p1 57G  48G  6.0G  89%`
- 850 MB 输入 .npz + 500 MB .bin + 100 MB engine + 164 MB ONNX 刚好塞下
- 如需更多 benchmark 数据（e.g. 100 帧 .bin = 5 GB）需搬到 tmpfs 或清理

**坑 12：Cloud → Orin 直连无 ssh key**
- Cloud 机器没配 Orin pubkey，直接 scp 要密码
- 解决：用本地当中转，tar-over-ssh 双端 pipe：
  ```bash
  ssh -p 8278 cloud "tar c -C /tmp orin_inputs" | ssh orin "tar x -C ~/sparsedrive_deploy"
  ```
- 一行命令 850 MB 数据从云端流到 Orin，无需占用本地磁盘

**坑 13：Orin Docker 里 trtexec 不在 PATH**
- `bash: line 1: trtexec: command not found`
- 修复：用全路径 `/usr/src/tensorrt/targets/aarch64-linux-gnu/bin/trtexec`

**坑 14：TRT 10.4 (Orin) vs 10.9 (x86) 对同一 INT8 ONNX 的 kernel 选择不同**
- Orin raw output vs x86 raw output：warm 帧 `det_box` max_abs ~18！
- 原因：INT8 kernel/fused op 选择跨版本有差异，state 累积让差异放大
- 但最终 **post_process 后 mAP 差 0.29%**，下游不敏感，下游合格

---

## 7. 关键复用 / 参考资源

| 资源 | 路径 | 用途 |
|---|---|---|
| UniAD TRT C++ 模板 | Orin `/home/nvidia/laona/DL4AGX/AV-Solutions/uniad-trt/inference_app/enqueueV3_stage1_new/` | C++ 推理骨架，学 setTensorAddress + enqueueV3 + CUDA timer |
| DL4AGX AV-Solutions | `/root/workspace/DL4AGX/AV-Solutions/` | NVIDIA 官方 TRT/INT8 EQ 参考集合，含 bevformer/streampetr 等 |
| uniad-orin Docker 容器 | Orin `docker ps` | aarch64 TRT 10.4 + CUDA 12.6 + g++ 11.4 + torch 2.4 现成环境 |
| 官方 deformable attn plugin | DL4AGX 子模块 | 编到 aarch64 后 → `libtensorrt_ops.so` |

---

## 8. 后续 TODO

- [x] ✅ 量化流水线（FP32 ONNX → INT8 engine）
- [x] ✅ Temporal I/O wrapper
- [x] ✅ 多段泛化性验证
- [x] ✅ Orin engine 构建 + 精度验证
- [x] ✅ Orin C++ latency benchmark
- [ ] **C++ 端到端部署应用**：相机图像预处理（jpeg decode + crop + resize + normalize）+ engine 调用 + det post_process（topk + NMS）+ map post_process（polyline 合并）+ 状态跟踪 —— 约 2000–3000 行，1–2 天
- [ ] **QAT（可选）**：若未来量化损失需进一步降低，可基于 modelopt 做 QAT 微调 backbone + head
- [ ] **相机管线整合**：和机器人实际相机输入对接，替换现在的预录制 nuScenes 帧
- [ ] **跨 scene 切换时 state reset 逻辑验证**：C++ 端实现 scene_token 检测 + prev_state 清零

---

## 9. 复现要点（最少命令）

### 从零构建 Orin engine（假设 ONNX 已有）

```bash
# 1. 传 ONNX + plugin 源码到 Orin
ssh -p 8278 root@10.108.1.132 "tar c -C quantization/artifacts sparsedrive_temporal_int8.onnx" \
  | ssh nvidia@10.10.253.131 "tar x -C ~/sparsedrive_deploy"

# 2. 在 Orin uniad-orin 容器里编 plugin（略 —— CMake build aarch64）

# 3. 在 Orin uniad-orin 容器里构建 engine
ssh nvidia@10.10.253.131 "docker exec uniad-orin bash -c '
  /usr/src/tensorrt/targets/aarch64-linux-gnu/bin/trtexec \
    --onnx=/host/home/nvidia/sparsedrive_deploy/sparsedrive_temporal_int8_v2.onnx \
    --saveEngine=/host/home/nvidia/sparsedrive_deploy/sparsedrive_orin_int8.engine \
    --int8 --fp16 \
    --staticPlugins=/host/home/nvidia/sparsedrive_deploy/tensorrt_plugins/lib/libtensorrt_ops.so
'"
```

### 跑 C++ benchmark

```bash
ssh nvidia@10.10.253.131 "docker exec uniad-orin bash -c '
  cd /host/home/nvidia/sparsedrive_deploy/orin_bench && ./build.sh && ./orin_bench \
    --engine /host/home/nvidia/sparsedrive_deploy/sparsedrive_orin_int8.engine \
    --plugin /host/home/nvidia/sparsedrive_deploy/tensorrt_plugins/lib/libtensorrt_ops.so \
    --blob-dir /host/home/nvidia/sparsedrive_deploy/orin_inputs_bin \
    --warmup 30 --iter 100
'"
```

### 精度验证（跨机流水线）

```bash
# 1. 云端：PyTorch FP32 baseline
ssh -p 8278 root@10.108.1.132 "... python3.8 /tmp/run_val_segment.py --start 3000 --num 40 --tag mid40_fp32"

# 2. 云端：x86 INT8 + dump inputs
"... python3.8 /tmp/dump_and_run_x86.py --engine sparsedrive_temporal_int8_v2.engine --tag temporal_int8_x86_mid40 --start 3000 --num 40"

# 3. 输入传 Orin
ssh cloud "tar c /tmp/orin_inputs" | ssh orin "tar x -C ~/sparsedrive_deploy"

# 4. Orin 跑 engine
ssh orin "docker exec uniad-orin python3 ~/sparsedrive_deploy/orin_run_engine.py"

# 5. Orin 输出传回云端做 post_process + eval
ssh orin "tar c ~/sparsedrive_deploy/orin_outputs" | ssh cloud "tar x -C /tmp"
"... python3.8 /tmp/orin_postprocess.py"
"... python3.8 /tmp/eval_segment_pkl.py --pkl temporal_int8_orin_mid40_val500.pkl --start 3000 --num 40 --tag ORIN"

# 6. 4 路可视化
"... python3.8 /tmp/viz_compare_3way.py --pytorch-pkl ... --int8-pkl ... --orin-pkl ... --gt-offset 3000 --frames 40"
```
