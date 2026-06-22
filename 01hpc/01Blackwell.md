# Blackwell 微架构与 GPGPU 大模型算子开发应用

> 适用对象：数据中心 NVIDIA Blackwell / Blackwell Ultra 类 GPU，以及面向 LLM 训练与推理的 CUDA/CUTLASS/CuTe/Triton/TensorRT-LLM/vLLM/FlashAttention/FlashMLA/DeepGEMM 类算子开发。

## 一、回答目标

本笔记解决三个问题：

1. 从芯片微架构视角解释 Blackwell 这类 GPGPU 的层次结构：GPU → Die → GPC → SM → Warp Scheduler → CUDA Core / Tensor Core / SFU / LDST / SMEM / TMEM / L2 / HBM / NVLink。
2. 从 kernel 编程视角说明高性能算子如何把数据流映射到 Blackwell：GMEM → TMA → SMEM → tcgen05/UMMA → TMEM → Epilogue → GMEM。
3. 从大模型工程视角说明 DeepSeek、ChatGPT 类 Transformer / MoE / MLA / Attention / GEMM / Norm / Decode / MoE dispatch 等工作负载为什么高度依赖 GPGPU 算子优化。

> 注意：ChatGPT 属于闭源商业模型服务，这里不假设能看到其真实 kernel 实现；只按公开大模型通用计算图和开源模型工程模式解释。DeepSeek-V3/R1 等公开权重/论文模型可以更直接地从 MLA、DeepSeekMoE、FP8、MoE routing、FlashMLA、GEMM、attention decode 等方向分析算子。

## 二、核心结论

1. **Blackwell 的关键变化不是“CUDA 线程变了”，而是 Tensor Core 数据通路变了。** Hopper 的 WGMMA 结果主要进入寄存器；Blackwell 数据中心 SM100 引入 tcgen05/UMMA 和 TMEM，使 MMA 累加结果进入 Tensor Memory，减少寄存器压力，并支持更大的异步 pipeline。
2. **高性能 LLM kernel 的本质是三级流水：搬运、计算、写回。** Producer warp 通过 TMA 把 A/B/KV tile 从 HBM/GMEM 搬到 SMEM；Consumer warp 发 tcgen05.mma；Epilogue warp 从 TMEM/SMEM 做转换、scale、激活、写回。
3. **LLM 的主耗时算子几乎都能归约为 GEMM、attention、reduction、scatter/gather、communication。** 开源模型如 DeepSeek-V3 的 MLA 和 DeepSeekMoE 使 attention KV-cache、MoE dispatch、grouped GEMM、all-to-all 通信变成核心优化对象。
4. **工程落地应优先使用库，再做定制 kernel。** 对标准 GEMM 先用 cuBLASLt/CUTLASS；对 attention 优先 FlashAttention/FlashInfer/TensorRT-LLM；对模型特有逻辑如 MLA、MoE routing、FP8 scale、expert grouped GEMM，再写 CUTLASS/CuTe/Triton/CUDA custom kernel。
5. **Blackwell 编译目标需要特别小心。** 如果使用 SM100 family-specific 特性，应使用 CUDA 12.9+ 引入的 `f` suffix 目标族，例如 `-gencode arch=compute_100f,code=sm_100`。使用架构/家族特定 PTX 后，兼容性与普通 PTX 不同。

## 三、Blackwell 微架构分层

### 3.1 总体层次

```text
Application / PyTorch / vLLM / TensorRT-LLM / DeepGEMM
        │
        ▼
CUDA Runtime / Driver / Stream / Graph / NCCL
        │
        ▼
GPU Device
 ├─ HBM3E / Global Memory
 ├─ L2 Cache / Fabric / NVLink / NVSwitch
 ├─ GPC / TPC / SM array
 │   └─ SM
 │      ├─ Warp Scheduler / Dispatch
 │      ├─ Register File
 │      ├─ CUDA Cores: FP32/INT32/FP16/BF16 等普通算术
 │      ├─ Tensor Cores 5th Gen: FP8/FP6/FP4/BF16/TF32/INT8 等矩阵计算
 │      ├─ TMEM: Tensor Core 专用中间结果存储
 │      ├─ Shared Memory / L1
 │      ├─ LD/ST + TMA path
 │      └─ SFU: exp/sin/rsqrt 等特殊函数
 └─ NVLink / PCIe / NVLink-C2C
```

### 3.2 层级职责表

| 层级 | 硬件/软件对象 | 面向算子的含义 | 大模型典型应用 |
|---|---|---|---|
| HBM / GMEM | 模型权重、激活、KV cache | 容量大、带宽高、延迟高 | 权重加载、KV cache 读写、activation tensor |
| L2 | 全 GPU 共享缓存 | 跨 SM 数据复用、写回聚合 | KV cache reuse、GEMM B 矩阵复用、MoE expert 权重复用 |
| SMEM/L1 | CTA 局部片上缓存 | tile 级复用，降低 GMEM 访问 | GEMM A/B tile、attention Q/K/V block、softmax 临时数据 |
| TMEM | Tensor Core 专用 memory | 保存 MMA accumulator，降低寄存器压力 | Blackwell tcgen05 GEMM/attention matmul accumulator |
| Register File | 线程私有寄存器 | scalar、指针、fragment、临时变量 | scale、偏移、mask、partial reduction |
| Tensor Core | 矩阵乘累加阵列 | LLM 最高吞吐来源 | QKV/MLP/Expert GEMM、attention QK/AV |
| CUDA Core | 标量/向量 ALU | 非矩阵逻辑、索引、激活、归约 | RMSNorm、RoPE、SwiGLU、top-k、routing |
| SFU | 特殊函数单元 | exp/rsqrt 等 | softmax exp、RMSNorm rsqrt、GELU/tanh 近似 |
| NVLink/NVSwitch/NCCL | GPU 间互联 | scale-up/scale-out 通信 | tensor parallel all-reduce、expert parallel all-to-all |

## 四、Blackwell SM100 面向算子的关键变化

### 4.1 从 Hopper WGMMA 到 Blackwell tcgen05/UMMA

```text
Hopper 典型路径：
GMEM -> TMA -> SMEM -> WGMMA -> Register accumulator -> Epilogue -> GMEM

Blackwell 数据中心 SM100 典型路径：
GMEM -> TMA -> SMEM -> tcgen05/UMMA -> TMEM accumulator -> Epilogue -> GMEM
```

| 对比项 | Hopper H100 | Blackwell SM100/Blackwell Ultra | 对 kernel 设计影响 |
|---|---|---|---|
| MMA 指令族 | `wgmma.mma_async` | `tcgen05.mma` / CUTLASS 中常称 UMMA | 需要新的 PTX/CUTLASS atom 和 pipeline |
| accumulator | 主要在寄存器 | TMEM | 降低寄存器压力，epilogue 需要读 TMEM |
| 数据搬运 | TMA + SMEM | TMA + SMEM + TMEM | producer/consumer/epilogue 分工更明确 |
| 低精度 | FP8/BF16/TF32 等 | FP8/FP6/FP4/NVFP4 + block scale | 量化 scale 和数据布局成为一等工程问题 |
| CTA 协作 | warp-group | CTA group / CTA pair 可参与 | cluster/CTA pair 调度更重要 |

### 4.2 Blackwell GEMM kernel 的标准数据流

```text
            ┌──────────────────────────────────────────────┐
            │ CTA tile: C[M_tile, N_tile]                   │
            └──────────────────────────────────────────────┘

Producer warp(s)
  GMEM A/B tensor map
       │ cp.async.bulk.tensor / TMA
       ▼
  SMEM stage[0..S-1]  ── full_barrier.arrive(bytes)
       │
       │ full_barrier.wait(stage, phase)
       ▼
Consumer warp(s)
  SMEM descriptor A/B
       │ tcgen05.mma / UMMA
       ▼
  TMEM accumulator
       │ tcgen05.commit / tmem_full barrier
       ▼
Epilogue warp(s)
  TMEM -> registers/SMEM_CD -> scale/activation/cast
       │ TMA store / vector store
       ▼
  GMEM D
```

### 4.3 Producer-Consumer-Barrier 时序

```text
时间 ───────────────────────────────────────────────────────────────▶

Stage 0:  TMA load A0/B0 ── arrive full[0] ── MMA S0 ── arrive empty[0]
Stage 1:                 TMA load A1/B1 ── arrive full[1] ── MMA S1 ── arrive empty[1]
Stage 2:                                  TMA load A2/B2 ── arrive full[2] ── MMA S2
Stage 3:                                                   TMA load A3/B3 ── arrive full[3]

Producer: wait empty[s] -> issue TMA -> arrive full[s]
Consumer: wait full[s]  -> issue UMMA -> arrive empty[s]
Epilogue: wait tmem_full -> read TMEM -> store D
```

| Barrier | 生产者 | 消费者 | 常见 bug |
|---|---|---|---|
| empty_barrier[s] | wait 后才能覆盖 SMEM stage | MMA 完成后 arrive | stage 被提前覆盖，导致 mismatch |
| full_barrier[s] | TMA 完成后 arrive/expect_tx | wait 后才能读 SMEM | expect_tx 字节数错误、phase 错误、死等 |
| tmem_full | MMA 完成后 commit/arrive | epilogue wait | epilogue 读到未完成 accumulator |
| cluster barrier | CTA pair/cluster 协作 | CTA 间同步 | rank/cta_group 配置错导致 hang |

## 五、LLM 算子如何映射到 Blackwell

### 5.1 Transformer Block 到 kernel 类型

```text
Input hidden [B, S, H]
        │
        ├─ RMSNorm / LayerNorm              -> reduction + vector op
        ├─ QKV Projection GEMM              -> GEMM / grouped GEMM
        ├─ RoPE                             -> elementwise + sin/cos
        ├─ Attention QK^T                   -> GEMM-like / FlashAttention tile
        ├─ Softmax                          -> row-wise max/sum/exp reduction
        ├─ Attention P·V                    -> GEMM-like / FlashAttention tile
        ├─ Output Projection GEMM           -> GEMM
        ├─ FFN / SwiGLU / GeGLU             -> GEMM + activation + GEMM
        └─ MoE Router/Experts               -> top-k + dispatch + grouped GEMM + combine
```

### 5.2 算子类别与优化抓手

| 算子 | 性能瓶颈 | Blackwell 关键硬件 | 工程优化方向 |
|---|---|---|---|
| Dense GEMM | Tensor Core 吞吐、SMEM bank conflict、epilogue | TMA、tcgen05、TMEM、FP8/FP4 | CUTLASS/CuTe SM100 kernel、warp specialization、TMA pipeline |
| Attention prefill | QK/AV GEMM + softmax 中间矩阵过大 | Tensor Core、SMEM、SFU、L2 | FlashAttention 分块，避免 materialize S×S attention |
| Attention decode | KV cache 带宽、batch 小、访存不连续 | L2/HBM、TMA、SFU | Paged KV cache、FlashInfer/FlashMLA、multi-query/group-query/MLA |
| RMSNorm/LayerNorm | memory-bound + reduction | CUDA Core、SFU、warp shuffle | 向量化 LD/ST、warp/block reduction、融合 residual/add |
| RoPE | memory-bound + trig | CUDA Core/SFU | 预计算 cos/sin、融合 Q/K 写回 |
| SwiGLU/GELU | elementwise + bandwidth | CUDA Core/SFU | 与 GEMM epilogue fusion |
| MoE router top-k | 小规模排序/选择 | CUDA Core、shared memory | block-level top-k、prefix sum、token packing |
| MoE dispatch/combine | scatter/gather + all-to-all | L2/HBM、NVLink/NCCL | expert parallel、permutation fusion、load balance |
| Expert GEMM | many small/irregular GEMM | Tensor Core、TMA、TMEM | grouped GEMM、persistent kernel、stream-K |
| AllReduce/AllToAll | 跨 GPU 通信 | NVLink/NVSwitch/SHARP/NCCL | overlap compute/communication、reduce-scatter、expert parallel |

## 六、DeepSeek / ChatGPT 类模型中的算子应用

### 6.1 DeepSeek-V3/R1 的硬件相关点

DeepSeek-V3 是 MoE 模型，论文公开描述其总参数约 671B，每 token 激活约 37B，并采用 MLA、DeepSeekMoE、辅助损失-free load balancing、多 token prediction、FP8 训练等设计。工程上这意味着：

1. **MLA 降低 KV-cache 压力**：decode 阶段不再只是标准 MHA/GQA 的 KV cache 访问，而是涉及 latent KV 表示的投影与恢复，对 attention kernel 的数据布局、cache 组织和投影融合提出要求。
2. **DeepSeekMoE 使 grouped GEMM 变成主战场**：每个 token 只激活部分 experts，带来 token dispatch、expert grouping、load balance、expert GEMM、combine 的全链路优化。
3. **FP8/低精度训练要求 scale 管理**：FP8/FP4/NVFP4 不只是换 dtype，还需要 per-tensor/per-block/per-channel scale，常常要把 scale load、dequant、MMA、requant 融入同一 kernel pipeline。

### 6.2 ChatGPT 类闭源服务的通用推断

不能假设 ChatGPT 内部真实模型结构、kernel 和部署策略，但从 Transformer/推理服务共性看，类似服务通常会高度依赖：

| 场景 | 典型 kernel | 关键优化目标 |
|---|---|---|
| Prompt prefill | FlashAttention、QKV/MLP GEMM | 高吞吐，充分占满 Tensor Core |
| 单 token decode | KV-cache attention、small GEMM | 低延迟，降低 HBM 访问与调度开销 |
| 长上下文 | PagedAttention/FlashAttention/MLA 类优化 | KV-cache 管理，L2 命中率，减少 O(S²) 中间态 |
| MoE 推理 | Router top-k、dispatch、grouped GEMM | expert 负载均衡，all-to-all 通信隐藏 |
| 量化推理 | FP8/INT8/FP4/NVFP4 GEMM | 精度、吞吐、scale fusion、memory footprint |
| 多 GPU serving | NCCL all-reduce/all-to-all | tensor parallel / expert parallel / pipeline parallel |

## 七、Blackwell 上开发大模型算子的工程路径

### 7.1 推荐落地流程

```text
Step 0: 明确模型子图
  PyTorch FX / torch.profiler / Nsight Systems 找热点 op

Step 1: 先试成熟库
  GEMM: cuBLASLt / CUTLASS / DeepGEMM
  Attention: FlashAttention / FlashInfer / TensorRT-LLM / FlashMLA
  Communication: NCCL

Step 2: 判断是否需要自定义 kernel
  - 模型结构特殊：MLA、MoE、稀疏 attention
  - shape 特殊：batch 小、N/K 极端、expert GEMM 很碎
  - fusion 需求：norm + residual + quant + GEMM epilogue

Step 3: 写 kernel
  - 简单 memory-bound：Triton 或 CUDA C++
  - Tensor Core GEMM-like：CUTLASS/CuTe SM100
  - 极限性能：inline PTX/SASS + 手写 descriptor/pipeline

Step 4: 正确性验证
  - 小 shape deterministic input
  - FP32 reference
  - ULP/relative error
  - debug buffer 替代 device printf

Step 5: 性能验证
  - Nsight Systems 看并发/通信重叠
  - Nsight Compute 看 Tensor Core 利用率、SMEM/L2/HBM 瓶颈、active warps
```

### 7.2 Blackwell GEMM/Attention kernel 伪代码

```cpp
// 伪代码：Blackwell SM100 风格 warp-specialized GEMM pipeline
__global__ void sm100_gemm_kernel(A, B, D, descA, descB, descD) {
  // 1. CTA/warp 角色划分
  int warp_id = threadIdx.x / 32;
  bool is_producer = (warp_id == 0);
  bool is_mma      = (warp_id == 1);
  bool is_epi      = (warp_id >= 2);

  // 2. 初始化 mbarrier / TMEM allocation / TMA descriptors
  init_mbarriers();
  tmem_addr = tcgen05_alloc(/*columns*/);

  for (int k_tile = 0; k_tile < K; k_tile += BK) {
    int s = (k_tile / BK) % NumStages;

    if (is_producer) {
      wait(empty_barrier[s]);
      // GMEM tensor -> SMEM stage[s]
      tma_load_async(descA, smem_A[s], coords_A(k_tile));
      tma_load_async(descB, smem_B[s], coords_B(k_tile));
      arrive_expect_tx(full_barrier[s], bytes_A + bytes_B);
    }

    if (is_mma) {
      wait(full_barrier[s]);
      // SMEM descriptor -> Tensor Core -> TMEM accumulator
      tcgen05_mma(tmem_addr, smem_desc_A[s], smem_desc_B[s], accumulate = k_tile != 0);
      arrive(empty_barrier[s]);
    }
  }

  if (is_epi) {
    wait(tmem_full);
    // TMEM accumulator -> cast/scale/activation -> GMEM D
    epilogue_store(tmem_addr, D, descD);
  }

  tcgen05_dealloc(tmem_addr);
}
```

### 7.3 编译与 profiling 示例

```bash
# CUDA 12.9+ Blackwell family-specific 目标示例
nvcc -O3 -std=c++17 \
  -gencode arch=compute_100f,code=sm_100 \
  -lineinfo \
  -Xptxas=-v \
  -o app main.cu

# Nsight Compute 示例：按 kernel 名过滤
ncu --target-processes all \
  --kernel-name regex:sm100_gemm_kernel \
  --set full \
  ./app
```

## 八、面向 SDC200/自研后端 porting 的映射建议

| Blackwell/CUDA 概念 | 自研后端可抽象为 | 移植检查点 |
|---|---|---|
| Warp/CTA/Cluster | wave/warp/workgroup/cluster | lane id、warp id、CTA rank 是否一致 |
| TMA tensor map | DMA descriptor/tensor descriptor | base、stride、boxDim、swizzle、边界填充 |
| SMEM stage | SRAM double/multi buffer | bank conflict、stage offset、对齐 |
| mbarrier expect_tx | transaction-count barrier | 字节数、phase、arrive/wait 次数 |
| tcgen05.mma | 自研 MMA/Matrix ISA | accumulator 位置、K step、layout、transpose |
| TMEM | accumulator SRAM/register file | 地址分配、生命周期、epilogue 读写 |
| epilogue | cast/scale/activation/store | vector store 宽度、对齐、尾块 mask |
| NVLink/NCCL | inter-chip/inter-tile fabric | all-reduce/all-to-all 与 compute overlap |

## 九、风险与调试清单

| 问题 | 典型现象 | 定位方法 | 修复方向 |
|---|---|---|---|
| TMA descriptor 错 | A/B tile 错位，K>小 shape 后 mismatch | dump SMEM stage 到 GMEM | 校验 stride/boxDim/swizzle/coords |
| Barrier phase 错 | hang 或偶发 mismatch | debug buffer 记录 phase/stage | full/empty 双 barrier phase 轮转 |
| expect_tx 字节数错 | mbar pending overflow / timeout | 记录每 stage bytes | A/B bytes 与 TMA transaction 一致 |
| TMEM 地址错 | 输出行/列重复、覆盖 | dump TMEM logical tile | tmem alloc columns、M/N mapping、accumulate flag |
| epilogue store 错 | GEMM core 对但 D 错 | TMEM->SMEM_CD->GMEM 分段验证 | store tile shape、vector width、mask |
| 低精度 scale 错 | 数值整体偏大/偏小或溢出 | dump scale tensor | per-block scale layout 与 MMA descriptor 对齐 |
| MoE load imbalance | 部分 expert 慢、GPU 空转 | per-expert token count histogram | router bias、capacity、grouped GEMM packing |
| decode KV cache 不连续 | 延迟高、L2 miss 高 | Nsight Compute L2/HBM 指标 | page/block layout、prefetch、batch 合并 |

## 十、参考资料

- NVIDIA Blackwell Architecture 官方页面：Blackwell 架构、208B transistors、dual-die、Transformer Engine、NVLink/NVL72。
- NVIDIA Technical Blog: Inside NVIDIA Blackwell Ultra，包含 SM、TMEM、Tensor Core、HBM、NVLink-C2C 等说明。
- NVIDIA CUDA 12.9 Blog: Blackwell family-specific architecture features，说明 `compute_100f` / `sm_100` 编译目标。
- NVIDIA PTX ISA 9.3 文档：tensor、cp.async.bulk.tensor、mbarrier 等 PTX 指令章节。
- NVIDIA CUTLASS Blackwell SM100 GEMMs 文档：tcgen05.mma、低精度类型、吞吐与 CUTLASS 支持。
- DeepSeek-V3 Technical Report：MLA、DeepSeekMoE、FP8、671B/37B、14.8T tokens、H800 GPU hours 等。


## reticle die 是什么？

**reticle die** 可以理解为：

> **受光刻机单次曝光视场大小限制，能一次曝光制造出来的最大级别芯片 die。**

更准确地说，NVIDIA Blackwell 官方描述是 **two reticle-limited dies**，也就是 **两个接近/达到光刻 reticle 尺寸限制的大 die**，通过高速 die-to-die 互连封装成一个逻辑上的统一 GPU。NVIDIA 官方 Blackwell 架构页写到：Blackwell GPU 由两个 reticle-limited dies 组成，通过 **10 TB/s chip-to-chip interconnect** 连接，对外表现为一个 unified single GPU。([NVIDIA][1])

---

## 1. reticle / die / reticle die 的关系

| 名词                      | 含义                         | 工程理解                                 |
| ----------------------- | -------------------------- | ------------------------------------ |
| **reticle**             | 光刻掩模版/曝光视场限制               | 光刻机一次能投影到晶圆上的最大图案区域                  |
| **die**                 | 晶圆切割后的单颗芯片裸片               | 一个 GPU 芯片、CPU 芯片或 chiplet            |
| **reticle-limited die** | 尺寸接近光刻曝光上限的 die            | 单个 die 已经做到很大，继续做大受光刻视场限制            |
| **dual reticle die**    | 两个 reticle-limited dies 组合 | Blackwell 这类 GPU 用两个大 die 拼成一个逻辑 GPU |

---

## 2. 为什么 Blackwell 要用两个 reticle-limited dies？

因为大模型训练/推理需要：

* 更多 SM；
* 更多 Tensor Core；
* 更大 L2；
* 更高 HBM 带宽；
* 更多晶体管；
* 更高 GEMM / Attention 吞吐。

但单个 monolithic die 不能无限变大，主要受限于：

```text
光刻 reticle limit
        │
        ▼
单次曝光面积有限
        │
        ▼
单颗 die 面积不能无限增大
        │
        ▼
继续堆 SM / Tensor Core / L2 需要多 die
        │
        ▼
用高速 die-to-die interconnect 把两个 die 连成一个 GPU
```

所以 Blackwell 的策略是：

```text
Reticle-limited Die 0        Reticle-limited Die 1
+-------------------+        +-------------------+
| SM / Tensor Core  |        | SM / Tensor Core  |
| L2 Slice          | <----> | L2 Slice          |
| HBM Interface     | NV-HBI | HBM Interface     |
+-------------------+        +-------------------+

对 CUDA 程序员：通常表现为一个统一 GPU device
```

NVIDIA 技术博客也描述 Blackwell Ultra 由两个 reticle-sized dies 构成，通过 NVIDIA High-Bandwidth Interface，也就是 **NV-HBI**，进行 die-to-die 连接。([NVIDIA Developer][2])

---

## 3. 对 CUDA / 算子开发有什么影响？

对普通 CUDA kernel 来说，你通常不需要显式关心“这个 CTA 跑在哪个 die 上”。它对外是一个统一 GPU。

但对高性能算子开发，尤其是 GEMM、Attention、MoE，有几个隐含影响：

| 影响点                          | 对算子工程的含义                                                |
| ---------------------------- | ------------------------------------------------------- |
| **跨 die 访问可能有拓扑差异**          | 极限性能优化时要关注 L2/HBM locality、SM 分布、block scheduling       |
| **L2 / HBM 资源可能物理分布在两个 die** | KV cache、权重 tile、persistent kernel 可能受 locality 影响      |
| **NCCL / NVLink 层次更复杂**      | 多 GPU、多节点训练通信需要考虑拓扑                                     |
| **单 GPU 逻辑统一**               | 大多数 CUDA API、PyTorch、cuBLASLt、CUTLASS 使用上仍按一个 device 看待 |
| **性能建模更复杂**                  | roofline 不只看一个 SM 或一个 HBM 通道，还要考虑 die-to-die 互连         |

---

## 4. 和 chiplet / MCM 的区别

可以粗略这样理解：

```text
chiplet / MCM 是封装组织方式
reticle-limited die 是 die 尺寸来源/制造约束
```

| 概念                      | 重点                            |
| ----------------------- | ----------------------------- |
| **chiplet**             | 多个小/中等 die 组合成一个系统            |
| **MCM**                 | Multi-Chip Module，多裸片封装       |
| **reticle-limited die** | 单个 die 已经接近光刻最大曝光面积           |
| **Blackwell dual-die**  | 两个超大 die 通过 NV-HBI 连成一个统一 GPU |

所以 Blackwell 不是简单“很多小 chiplet 拼起来”，而是 **两个非常大的 reticle-limited GPU die 拼成一个统一 GPU**。

---

## 5. 一句话总结

**reticle die / reticle-limited die** 就是：

> 单个 die 的尺寸已经接近光刻机一次曝光能制造的最大面积；Blackwell 用两个这种大 die，通过 10 TB/s 级别 NV-HBI 互连，封装成一个对 CUDA 程序基本透明的统一 GPU。

[1]: https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/?utm_source=chatgpt.com "NVIDIA Blackwell Architecture"
[2]: https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/?utm_source=chatgpt.com "Inside NVIDIA Blackwell Ultra: The Chip Powering the AI ..."
