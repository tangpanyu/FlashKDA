# FlashKDA Forward Kernel 读代码指南

这份文档解释当前 `csrc/smxx` 下 forward 实现的主干逻辑。重点不是逐行翻译 CUTE/CUTLASS 模板，而是说明每个 kernel 在做什么、数据如何流动、哪些中间量值得盯住。

## 1. 文件入口

forward 路径由这几个文件组成：

| 文件 | 作用 |
|---|---|
| `csrc/flash_kda.cpp` | PyTorch 入口，检查 shape/dtype，整理指针，分发不同 state/varlen 模板 |
| `csrc/fwd.h` | `launch_fwd` 声明 |
| `csrc/smxx/fwd_launch.cu` | 创建 CUTE tensor/TMA descriptor，连续 launch K1 和 K2 |
| `csrc/smxx/fwd_kernel1.cuh` | K1: tile 级预处理，生成 workspace |
| `csrc/smxx/fwd_kernel2.cuh` | K2: sequence/head 级 recurrence，写 out/final_state |
| `csrc/smxx/utils.cuh` | 近似函数、pipeline helper、小 MMA、Neumann inverse、state dtype 转换 |

当前实现固定 `D = 128`，`CHUNK = 16`。这里的 `D` 同时对应 K/V 维度，因为 API 要求 `K = V = 128`。

## 2. 总体数据流

Python 侧传入：

| 张量 | shape | dtype |
|---|---|---|
| `q, k, v, g` | `[B, T, H, D]` | bf16 |
| `beta` | `[B, T, H]` | bf16 |
| `A_log` | `[H]` | fp32 |
| `dt_bias` | `[H, D]` | fp32 |
| `initial_state/final_state` | `[N, H, D, D]` | bf16/fp32 |

`flash_kda.cpp` 把 `[B, T, H, D]` reshape 为逻辑上的 `[T_total, H, D]`，然后 `fwd_launch.cu` 用 CUTE layout 把它按 `[H, T_total, D]` 访问：

$$
\text{offset}(h,t,d) = t \cdot H \cdot D + h \cdot D + d
$$

`beta` 会先转置成 `[H, T_total]` 的连续内存，因为两个 kernel 都按 `head` 和连续 token tile 读取 beta。这样 1D TMA 更简单。

整个 forward 分两段：

1. K1: 每个 `(head, chunk)` 一个 CTA，把 q/k/g/beta 预处理成 workspace。
2. K2: 每个 `(sequence, head)` 一个 CTA，沿 chunk 顺序做 recurrent state update 并写输出。

这种拆法的原因是并行轴不同。K1 有大量 token/chunk 并行；K2 必须沿 sequence chunk 顺序递推，但不同 sequence/head 之间仍并行。

## 3. Workspace 布局

K1 产生 K2 要用的中间量，全部写进用户传入的 `workspace`。每个 `(head, tile)` 有 6 段：

| 名称 | shape | dtype | bytes |
|---|---:|---|---:|
| `k_decayed` | `[CHUNK, D]` | bf16 | 4096 |
| `q_decayed` | `[CHUNK, D]` | bf16 | 4096 |
| `k_restored` | `[CHUNK, D]` | bf16 | 4096 |
| `g_total` | `[D]` | fp32 | 512 |
| `INV` | `[CHUNK, CHUNK]` | bf16 | 512 |
| `Mqk` | `[CHUNK, CHUNK]` | bf16 | 512 |

单个 head-tile 总大小：

$$
3 \cdot 16 \cdot 128 \cdot 2 + 128 \cdot 4 + 2 \cdot 16 \cdot 16 \cdot 2 = 13824 \text{ bytes}
$$

`fwd_launch.cu` 里把这 6 段拆成不同 typed pointer，K1 store，K2 load。`get_workspace_size()` 给的是 upper bound，varlen 时每个 sequence 至多多算一个 tile。

## 4. `launch_fwd`: 两个 kernel 的调度器

`launch_fwd<D, HasStateIn, HasStateOut, StateFP32, IsVarlen>` 是 C++ 模板分发后的真正 CUDA launch 入口。

### 4.1 K1 launch

K1 kernel 名为 `_flash_kda_fwd_prepare`：

```cpp
dim3 grid_k1(total_tiles, H);
dim3 block_k1(256);
```

含义：

- `blockIdx.x`: 全局 tile id。
- `blockIdx.y`: head id。
- 每个 CTA 处理一个 head 下的一个 chunk。
- varlen 时，kernel 内部用 `cu_seqlens` 线性扫描，把全局 tile id 映射到 `(seq_idx, local_t)`。

K1 输出 workspace，不直接写最终 `out`。

### 4.2 K2 launch

K2 kernel 名为 `_flash_kda_fwd_recurrence`：

```cpp
dim3 grid_k2(N, H);
dim3 block_k2(32 * 2 + 128); // 192 threads
```

含义：

- `blockIdx.x`: sequence id。
- `blockIdx.y`: head id。
- 每个 CTA 负责一个完整 sequence-head 的 recurrence。
- 192 threads 被切成 6 个 warp：
  - 4 个 MMA warp，做矩阵乘和 state update。
  - 1 个 LOAD warp，发 TMA load。
  - 1 个 STORE warp，发 TMA/manual store。

K2 用 3-stage input pipeline 和 2-stage output pipeline，把 TMA load/store 与 MMA 计算重叠。

## 5. K1: `_flash_kda_fwd_prepare`

K1 可以理解为“把一个 16-token chunk 变成 K2 更容易消费的块内矩阵和衰减因子”。

### 5.1 读入 q/k/beta/g/dt_bias

每个 CTA 先定位当前 chunk 的 `[16, 128]` 数据。thread 0 发 TMA load：

- `q`: `[CHUNK, D]`
- `k`: `[CHUNK, D]`
- `g`: `[CHUNK, D]`
- `beta`: 1D `[32]` staging buffer，实际只用当前 16 个 token；因为做了 8 对齐。
- `dt_bias`: `[D]`

尾块如果不足 16 token，后续会用 `actual_len` 做保护，并把 tail 的 `k` 清零。

### 5.2 q/k L2 normalize

每行 token 的 q/k 都做 L2 normalize。每行 128 个元素由 16 个线程处理，每线程 8 个元素，warp 内 shuffle 做 reduce：

$$
\hat q_t = \frac{q_t}{\sqrt{\sum_d q_{t,d}^2 + 10^{-6}}}
$$

$$
\hat k_t = \frac{k_t}{\sqrt{\sum_d k_{t,d}^2 + 10^{-6}}}
$$

这里和 README 里的 `use_qk_l2norm_in_kernel=True` 对应。

### 5.3 gate activation 和 cumsum

代码把 gate 转成 base-2 log 空间，后面用 `ex2.approx.ftz.f32`。对每个 token 和 feature：

$$
g'_{t,d} =
\log_2(e) \cdot lower\_bound \cdot
\sigma\left(\exp(A\_log_h) \cdot (g_{t,h,d} + dt\_bias_{h,d})\right)
$$

其中 sigmoid 由 `tanh.approx.f32` 实现：

$$
\sigma(x) = \frac{1 + \tanh(x/2)}{2}
$$

然后沿 chunk 内 token 做 prefix sum：

$$
c_{t,d} = \sum_{\tau=0}^{t} g'_{\tau,d}
$$

并保存最后一个 token 的总衰减：

$$
G_d = 2^{c_{15,d}}
$$

若是尾块，不存在的 token 对应 `g' = 0`，这样不会污染后续计算。

### 5.4 生成 `q_decayed/k_decayed/k_restored`

K1 接着把 normalize 后的 q/k 变换成 K2 需要的三种形式：

$$
q\_decayed_{t,d} = scale \cdot \hat q_{t,d} \cdot 2^{c_{t,d}}
$$

$$
k\_decayed_{t,d} = \hat k_{t,d} \cdot 2^{c_{t,d}}
$$

$$
k\_inv_{t,d} = \hat k_{t,d} \cdot 2^{-c_{t,d}}
$$

$$
k\_restored_{t,d} = \hat k_{t,d} \cdot 2^{c_{15,d} - c_{t,d}}
$$

注意 `k_inv` 只在 K1 shared memory 中临时使用，不写 workspace。真正写出的第三个 `[CHUNK, D]` 是 `k_restored`。

### 5.5 构造块内矩阵 `L` 和 `Mqk`

K1 用 1 warp 做 `k_decayed @ k_inv^T`，另 1 warp 做 `q_decayed @ k_inv^T`：

$$
L^{raw}_{i,j} = \sum_d k\_decayed_{i,d} \cdot k\_inv_{j,d}
$$

$$
Mqk^{raw}_{i,j} = \sum_d q\_decayed_{i,d} \cdot k\_inv_{j,d}
$$

随后：

- `L` 只保留严格下三角，并乘以 `sigmoid(beta_i)`。
- `Mqk` 保留下三角含对角，清掉上三角。

也就是：

$$
L_{i,j} =
\begin{cases}
L^{raw}_{i,j} \cdot \sigma(\beta_i), & i > j \\
0, & i \le j
\end{cases}
$$

$$
Mqk_{i,j} =
\begin{cases}
Mqk^{raw}_{i,j}, & i \ge j \\
0, & i < j
\end{cases}
$$

这里的三角结构来自 chunk 内的 causal 依赖：当前 token 只能依赖自己和之前 token。

### 5.6 计算 `INV`

K1 需要的是块内 triangular solve 的逆矩阵。`utils.cuh::neumann_inv_fused_1warp` 做的是：

$$
INV = (I + L)^{-1}
$$

由于 `L` 是 16x16 严格下三角矩阵，满足 $L^{16}=0$，所以可以用有限 Neumann 展开：

$$
(I+L)^{-1} = I - L + L^2 - L^3 + \cdots - L^{15}
$$

代码里的实现方式是先构造：

$$
INV_0 = I - L
$$

然后依次融合：

$$
INV = (I-L)(I+L^2)(I+L^4)(I+L^8)
$$

展开后正好是到 15 次的交错级数。因为矩阵只有 16x16，这个过程由单 warp 用 fp16 MMA 完成，最后转 bf16 存到 workspace。

### 5.7 K1 输出

K1 最后把这些结果 TMA store 到 workspace：

- `k_decayed`
- `q_decayed`
- `k_restored`
- `g_total`
- `INV`
- `Mqk`

这就是 K2 每个 chunk 的输入包。

## 6. K2: `_flash_kda_fwd_recurrence`

K2 可以理解为“拿 K1 处理好的 chunk 包，按时间顺序更新 recurrent state，并生成 output”。

### 6.1 state 初始化

每个 K2 CTA 对应一个 `(seq_idx, head_idx)`，shared memory 里维护一个：

$$
S \in \mathbb{R}^{D \times D}
$$

代码里叫 `state_acc`。

初始化分三种：

- `HasStateIn && !StateFP32`: TMA load bf16 state 到 `state_acc`。
- `HasStateIn && StateFP32`: TMA load fp32 state 到临时 shared buffer，再转 bf16 到 `state_acc`。
- 无 initial state: 把 `state_acc` 清零。

虽然外部允许 fp32 state，K2 计算主路径仍把 on-chip state 存成 bf16，MMA accumulation 走 fp32。

### 6.2 warp specialization 和 pipeline

K2 的 6 个 warp 分工：

| warp | 角色 |
|---:|---|
| 0-3 | MMA 计算 |
| 4 | TMA load |
| 5 | TMA/manual store |

LOAD warp 对每个 chunk 读入：

- `v`
- `beta`
- K1 workspace 的 `k_decayed/q_decayed/k_restored/g_total/INV/Mqk`

MMA warps 通过 `load_pipeline.consumer_wait()` 等待当前 stage ready；STORE warp 通过 `store_pipeline.consumer_wait()` 等待 output stage ready。

### 6.3 每个 chunk 的数学步骤

设当前 chunk 的 value 为：

$$
V \in \mathbb{R}^{16 \times D}
$$

K1 传来的矩阵为：

$$
K_d = k\_decayed,\quad Q_d = q\_decayed,\quad K_r = k\_restored
$$

K2 的核心 fused 计算对应以下步骤。

第一步，用旧 state 计算已有历史对当前 chunk 的影响：

$$
U_0 = K_d S
$$

$$
O_0 = Q_d S
$$

代码里 `u_acc` 存 $U_0$，`out_acc` 存 $O_0$。每个 MMA warp 负责 output feature 的 32 列，也就是两个 16x16 column block。

第二步，构造 delta 更新量。代码做：

$$
U_1 = (V - U_0) \odot \sigma(\beta)
$$

这里 `beta` 按 token 广播到 feature 维。

第三步，解 chunk 内部依赖：

$$
U = INV \cdot U_1
$$

这一步使用 K1 的 `INV = (I+L)^{-1}`。`U` 全程尽量留在寄存器里，并用 `MOVM_T` 做寄存器级转置，避免写回 shared memory。

第四步，生成最终输出：

$$
O = O_0 + Mqk \cdot U
$$

然后把 `O` store 到 output staging shared memory，交给 STORE warp 写回 `out`。

第五步，更新 recurrent state：

$$
S_{new,d,:} = G_d \cdot S_{old,d,:} + \sum_{t=0}^{15} K_{r,t,d} \cdot U_{t,:}
$$

矩阵形式可以写成：

$$
S_{new} = \operatorname{diag}(G) S_{old} + K_r^T U
$$

这里 $G$ 就是 K1 写出的 `g_total`，即每个 feature 在整个 chunk 上的总衰减 $2^{c_{15,d}}$。

### 6.4 out 和 final_state 写回

STORE warp 写 out 时分两种：

- 满 16 token 的 chunk: TMA store `[CHUNK, D]`。
- 尾块: 手写 loop，只写 `actual_len` 行，避免越界覆盖下一条 sequence。

final state 写回也分 bf16/fp32：

- bf16: 直接 TMA store `state_acc`。
- fp32: 先全 CTA 同步，把 bf16 shared state 转 fp32 到临时 buffer，再 TMA store。

## 7. `utils.cuh` 里值得看的 helper

### 7.1 近似函数

`ex2_approx_ftz_f32()` 使用 PTX `ex2.approx.ftz.f32`。因为 K1 已经把 gate 转成 base-2 log，所以后续指数都可以走 `ex2`。

`sigmoid_tanh_approx_f32()` 使用：

$$
\sigma(x) = 0.5 \cdot \tanh(x/2) + 0.5
$$

底层是 `tanh.approx.f32`。

### 7.2 小矩阵 MMA

`mma_m16n16_bf16bf16bf16_1warp()` 和 `mma_m16n16_bf16bf16fp16_1warp()` 是 K1 构造 `Mqk`、`L` 时用的小 GEMM helper。

关键点：

- 输入是 shared memory 里的 CUTE tensor。
- 一个 warp 完成一个 16x16 tile。
- accumulation 是 fp32，store 时转 bf16/fp16。

### 7.3 Neumann inverse

`neumann_inv_fused_1warp()` 是 K1 的块内 inverse 核心。它不调用通用矩阵求逆，而是利用 16x16 严格三角矩阵 nilpotent 的性质，用 3 轮平方展开完成：

$$
(I-L)(I+L^2)(I+L^4)(I+L^8)
$$

### 7.4 state dtype conversion

`smem_cvt_fp32_to_bf16()` 和 `smem_cvt_bf16_to_fp32()` 只在外部 state dtype 是 fp32 时出现。它们按 8x8 atom 遍历 shared memory，使 fp32 state layout 和 bf16 state layout 对齐。

## 8. 建议的读代码顺序

第一次看不要从 CUTE layout 模板开始钻。推荐顺序：

1. 看 `flash_kda.cpp` 的 shape/dtype 检查和 `DISPATCH_STATE`，明确有哪些模板分支。
2. 看 `fwd_launch.cu` 的 workspace pointer 切分和两个 kernel launch。
3. 看 `fwd_kernel1.cuh` 的注释块顺序：TMA load -> normalize -> gate cumsum -> decay_apply -> `L/Mqk` -> `INV` -> workspace store。
4. 看 `fwd_kernel2.cuh` 的 Phase 1-6 注释，把每一相和上面的数学式对应起来。
5. 最后再看 `K1Layouts/K2Layouts` 和 CUTE copy atom，理解 shared memory swizzle 和 TMA layout。

## 9. 常见坑点

### 9.1 `g_total` 不是 raw sum

K1 中 `g_total` 先是 base-2 log prefix 的最后值，随后被改写成：

$$
g\_total_d = 2^{c_{15,d}}
$$

所以 K2 读到的 `g_total` 已经是乘法衰减因子，不是 log。

### 9.2 `beta` 是 pre-activation

API 传入的 `beta` 是 logits。K1 和 K2 内部都会调用 sigmoid 近似。不要在外部提前 sigmoid，否则会重复激活。

### 9.3 `beta_smem_offset`

TMA 读 beta 时按 8 对齐：

```cpp
int beta_aligned = beta_linear & ~7;
```

所以真正访问当前 token beta 时要加：

```cpp
int beta_smem_offset = beta_linear & 7;
```

这个 offset 在 K1/K2 都有。

### 9.4 K2 的 `state_acc` 是 bf16 存储、fp32 accumulate

shared memory 中的 state 是 bf16。MMA 的 accumulator 是 fp32，但每轮更新后会把结果写回 bf16 shared state。这是性能和精度的折中。

### 9.5 尾块 store 不能 TMA

K2 对尾块 out 使用手写 loop，是为了避免 TMA store 一个完整 16 行时覆盖后面的 sequence/token。

### 9.6 `BLOCK_LEVEL_K1/K2`

`utils.cuh` 默认：

```cpp
#define BLOCK_LEVEL_K1 1
#define BLOCK_LEVEL_K2 1
```

`fwd_launch.cu` 用 `#if BLOCK_LEVEL_K1 >= 0` 和 `#if BLOCK_LEVEL_K2 >= 0` 控制 kernel launch。调试时可以通过宏控制某一段是否启用，但要注意 K2 依赖 K1 workspace。

## 10. 一句话总结

K1 把每个 16-token chunk 转成“已经带 decay 的 q/k、块内 triangular solve 需要的 `INV/Mqk`、以及跨 chunk state 衰减因子”；K2 沿 sequence 消费这些 chunk 包，用 warp-specialized pipeline 完成 delta recurrence、输出和 final state 更新。
