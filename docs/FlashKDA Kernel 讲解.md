# FlashKDA Kernel 讲解

本文参考 `docs/KDA Kernel 简化版.md` 的算法主线和 `docs/20260420-flashkda-v1-deep-dive.md` 的设计取舍，解释当前 `csrc/smxx` 里的 forward kernel 是怎么把 KDA chunk 公式落到 CUDA 实现里的。

阅读时建议先记住一句话：

```text
K1 把每个 16-token chunk 预处理成 K2 好消费的矩阵包；
K2 沿 sequence 逐 chunk 做 delta recurrence，更新 state 并写 output。
```

当前实现固定：

```text
CHUNK = 16
D = K = V = 128
```

代码入口：

| 文件 | 角色 |
|---|---|
| `csrc/flash_kda.cpp` | PyTorch 扩展入口，检查 shape/dtype，选择模板分支 |
| `csrc/smxx/fwd_launch.cu` | 创建 CUTE layout/TMA descriptor，launch K1/K2 |
| `csrc/smxx/fwd_kernel1.cuh` | K1: chunk 预处理，写 workspace |
| `csrc/smxx/fwd_kernel2.cuh` | K2: chunk recurrence，写 out/final_state |
| `csrc/smxx/utils.cuh` | 近似函数、pipeline helper、小 MMA、Neumann inverse、state 转换 |

## 1. 从 KDA 单步到 chunk 公式

对一个 token，KDA 的 recurrent state 是固定大小矩阵：

$$
S_t\in\mathbb{R}^{d_k\times d_v}
$$

单步更新可以按“先 decay 旧 state，再用 delta rule 擦写”理解：

$$
\bar S_t=\operatorname{Diag}(\exp(g_t))S_{t-1}
$$

$$
v_{\Delta,t}=\beta_t(v_t-\bar S_t^\top k_t)
$$

$$
S_t=\bar S_t+k_tv_{\Delta,t}^\top
$$

$$
o_t=S_t^\top q_t
$$

KDA 和 GDN 的差异在 decay：GDN 是 scalar decay，KDA 是 key/channel-wise vector decay。所以 KDA 的 score 不能简单写成普通 $QK^\top$ 乘一个 scalar mask，而必须把 decay 放进 dot product 里面。

对一个 chunk，令 chunk size 为 $C$，这里实际 $C=16$。把 token 堆成矩阵：

$$
Q,K\in\mathbb{R}^{C\times d_k},\qquad V,O,D\in\mathbb{R}^{C\times d_v}
$$

raw gate 做 cumsum 后得到累计 log decay：

$$
g_r=\sum_{j=0}^{r}g^{raw}_j
$$

并定义：

$$
\Gamma_r=\exp(g_r)\in\mathbb{R}^{d_k}
$$

于是可以构造四个带 decay 的矩阵：

$$
Q_\gamma=\Gamma\odot Q
$$

$$
K_\gamma=\Gamma\odot K
$$

$$
K_{\gamma^{-1}}=K/\Gamma
$$

$$
K_{\text{right}}=(\Gamma_C/\Gamma)\odot K
$$

它们在当前 kernel 里的名字基本对应：

| 简化版符号 | 当前代码/workspace 名 | 含义 |
|---|---|---|
| $Q_\gamma$ | `q_decayed` | 带累计 decay 和 scale 的 q |
| $K_\gamma$ | `k_decayed` | 带累计 decay 的 k |
| $K_{\gamma^{-1}}$ | `k_inv` | K1 内部临时量，不写 workspace |
| $K_{\text{right}}$ | `k_restored` | 当前 chunk 写入传到 chunk 末尾后的 k |
| $\Gamma_C$ | `g_total` | chunk 总 decay，K2 用来 decay 旧 state |

注意当前实现把指数换成 base-2。K1 中保存的是 $2^{c_{t,d}}$，数学语义等价于上面的 $\exp(g)$，只是为了使用更快的 `ex2.approx.ftz.f32`。

## 2. FlashKDA 为什么拆成 K1/K2

deep-dive 中的核心设计是把 forward 切成两个 kernel：

```text
K1: token/chunk 并行，grid = [total_tiles, H]
K2: sequence/head 并行，grid = [N, H]
```

原因是两段天然并行度不同：

- K1 的每个 chunk 可以独立做 gate、norm、score 和 inverse，所以并行度高。
- K2 必须按 chunk 顺序更新同一个 recurrent state，所以每个 sequence/head 内部有时间依赖。

如果强行塞进单个 kernel，K1 的高并行 token 工作会被 K2 的低并行 recurrence 牵制。当前拆分后，K1 充分铺开，K2 专注顺序 recurrence。

## 3. `flash_kda.cpp`: 入口做了什么

Python 入口传入：

| 张量 | shape | dtype |
|---|---|---|
| `q, k, v, g` | `[B, T, H, D]` | bf16 |
| `beta` | `[B, T, H]` | bf16 |
| `A_log` | `[H]` | fp32 |
| `dt_bias` | `[H, D]` | fp32 |
| `initial_state/final_state` | `[N, H, D, D]` | bf16/fp32 |
| `cu_seqlens` | `[N+1]` | int64 |

入口主要做四件事：

1. 检查 CUDA、contiguous、dtype、shape。
2. 把 `[B,T,H,D]` reshape 成逻辑上的 `[T_total,H,D]`。
3. 把 `beta` 从 `[T_total,H]` 转置成 `[H,T_total]` 连续内存，方便两个 kernel 做 1D TMA。
4. 根据 `initial_state/final_state/cu_seqlens` 选择模板：

```cpp
launch_fwd<128, HasStateIn, HasStateOut, StateFP32, IsVarlen>(...)
```

`lower_bound` 会先乘以 $\log_2(e)$：

$$
gate\_scale=lower\_bound\cdot \log_2(e)
$$

这样 K1 中 gate activation 后的累计值就是 base-2 log decay，后续可以直接调用 `ex2.approx.ftz.f32`。

## 4. `fwd_launch.cu`: TMA descriptor 和 workspace

`launch_fwd` 里先建立逻辑 layout。原始 tensor 虽然内存是 `[T_total,H,D]`，但 CUTE 访问时使用 `[H,T_total,D]`：

$$
\text{offset}(h,t,d)=t\cdot H\cdot D+h\cdot D+d
$$

workspace 被切成 6 段，每个 `(head,tile)` 都有一份：

| workspace 段 | shape | dtype | 用途 |
|---|---|---|---|
| `ws_kd` | `[CHUNK,D]` | bf16 | `k_decayed` |
| `ws_qd` | `[CHUNK,D]` | bf16 | `q_decayed` |
| `ws_kr` | `[CHUNK,D]` | bf16 | `k_restored` |
| `ws_gt` | `[D]` | fp32 | `g_total` |
| `ws_inv` | `[CHUNK,CHUNK]` | bf16 | chunk 内 triangular inverse |
| `ws_mqk` | `[CHUNK,CHUNK]` | bf16 | chunk 内 query-key 读取矩阵 |

每个 head-tile 的 bytes 为：

$$
3\cdot C\cdot D\cdot 2+D\cdot 4+2\cdot C\cdot C\cdot 2
$$

当前 $C=16,D=128$，所以：

$$
3\cdot16\cdot128\cdot2+128\cdot4+2\cdot16\cdot16\cdot2=13824
$$

K1 只写 workspace，K2 再读 workspace。`fwd_launch.cu` 也会为 state 构造 TMA descriptor；如果外部 state 是 fp32，就用 fp32 layout 读写，中间在 shared memory 内转成 bf16 计算。

## 5. K1: `_flash_kda_fwd_prepare`

K1 的目标是把一个 chunk 变成：

```text
k_decayed, q_decayed, k_restored, g_total, INV, Mqk
```

也就是 K2 消费的“chunk 矩阵包”。

K1 launch 参数：

```cpp
dim3 grid_k1(total_tiles, H);
dim3 block_k1(256);
```

每个 CTA 处理一个 `(tile, head)`。

### 5.1 tile 定位

非 varlen 时：

$$
seq\_idx=\lfloor global\_tile/tiles\_per\_seq\rfloor
$$

$$
local\_t=global\_tile-seq\_idx\cdot tiles\_per\_seq
$$

varlen 时没有预先传 tile offset 表，K1 内部线性扫描 `cu_seqlens`，把 `global_tile_idx` 映射到 `(seq_idx, local_t)`。

### 5.2 TMA load

thread 0 发 TMA load，把当前 chunk 搬进 shared memory：

- `q`: `[16,128]`
- `k`: `[16,128]`
- `g`: `[16,128]`
- `beta`: 1D buffer，按 8 对齐读取
- `dt_bias`: `[128]`

`beta` 有一个容易忽略的 offset。TMA 地址按 8 对齐：

```cpp
int beta_aligned = beta_linear & ~7;
```

实际 token 的起点再用：

```cpp
int beta_smem_offset = beta_linear & 7;
```

### 5.3 q/k L2 norm

K1 内部直接做 q/k L2 normalization。每行 128 个元素，16 个线程处理一行，每线程 8 个元素，warp 内 shuffle reduce：

$$
\hat q_t=\frac{q_t}{\sqrt{\sum_d q_{t,d}^2+10^{-6}}}
$$

$$
\hat k_t=\frac{k_t}{\sqrt{\sum_d k_{t,d}^2+10^{-6}}}
$$

这对应 FLA 调用里的 `use_qk_l2norm_in_kernel=True`。

### 5.4 gate activation 和 cumsum

K1 将 raw `g` 和 `dt_bias` 合并，并加上 `A_log`：

$$
x_{t,d}=\exp(A\_log_h)\cdot(g^{raw}_{t,h,d}+dt\_bias_{h,d})
$$

`beta` 和 `g` 都是 pre-activation。`g` 的激活为：

$$
\ell_{t,d}=lower\_bound\cdot\sigma(x_{t,d})
$$

由于代码用 base-2：

$$
\tilde\ell_{t,d}=\log_2(e)\cdot lower\_bound\cdot\sigma(x_{t,d})
$$

其中 sigmoid 使用 `tanh.approx.f32`：

$$
\sigma(x)=0.5\cdot\tanh(0.5x)+0.5
$$

然后沿 chunk 内 token 做 prefix sum：

$$
c_{t,d}=\sum_{i=0}^{t}\tilde\ell_{i,d}
$$

最后保存：

$$
G_d=2^{c_{C-1,d}}
$$

代码中 `g_total` 最开始保存 $c_{C-1,d}$，随后原地改写成 $G_d$。因此 K2 读到的 `g_total` 已经是乘法衰减因子，不是 log。

### 5.5 decay_apply

K1 根据累计 decay 生成 K2 所需的三个 `[C,D]` 矩阵：

$$
q\_decayed_{t,d}=scale\cdot\hat q_{t,d}\cdot2^{c_{t,d}}
$$

$$
k\_decayed_{t,d}=\hat k_{t,d}\cdot2^{c_{t,d}}
$$

$$
k\_inv_{t,d}=\hat k_{t,d}\cdot2^{-c_{t,d}}
$$

$$
k\_restored_{t,d}=\hat k_{t,d}\cdot2^{c_{C-1,d}-c_{t,d}}
$$

其中 `k_inv` 只在 K1 shared memory 内临时使用，真正写进 workspace 的是 `k_decayed/q_decayed/k_restored`。

对照简化版：

```text
q_decayed   = Q_gamma
k_decayed   = K_gamma
k_inv       = K_gamma^{-1}
k_restored  = K_right
```

### 5.6 构造 `L` 和 `Mqk`

K1 用两个 warp 做 16x16 小 GEMM：

$$
B=K_\gamma K_{\gamma^{-1}}^\top
$$

$$
E=Q_\gamma K_{\gamma^{-1}}^\top
$$

代码里对应：

```text
L_raw  = k_decayed @ k_inv.T
Mqkraw = q_decayed @ k_inv.T
```

然后应用 causal 三角结构：

$$
L_{i,j}=
\begin{cases}
\sigma(\beta_i)B_{i,j},& i>j\\
0,& i\le j
\end{cases}
$$

$$
Mqk_{i,j}=
\begin{cases}
E_{i,j},& i\ge j\\
0,& i<j
\end{cases}
$$

这一步把简化版中的：

$$
\operatorname{StrictTril}(\operatorname{Diag}(\beta)B)
$$

放进了 `L`，并把：

$$
\operatorname{Tril}(Q_\gamma K_{\gamma^{-1}}^\top)
$$

放进了 `Mqk`。

### 5.7 `INV`: chunk 内三角解

简化版里有效写入满足：

$$
D=
\left[I+\operatorname{StrictTril}(\operatorname{Diag}(\beta)B)\right]^{-1}
\operatorname{Diag}(\beta)(V-K_\gamma S_0)
$$

当前代码将：

$$
L=\operatorname{StrictTril}(\operatorname{Diag}(\beta)B)
$$

于是 K1 只需要预计算：

$$
INV=(I+L)^{-1}
$$

因为 $L$ 是 $16\times16$ 严格下三角矩阵，所以 $L^{16}=0$，可以用有限 Neumann 展开：

$$
(I+L)^{-1}=I-L+L^2-L^3+\cdots-L^{15}
$$

`utils.cuh::neumann_inv_fused_1warp` 实际用下面的因式分解：

$$
INV=(I-L)(I+L^2)(I+L^4)(I+L^8)
$$

这正好展开到 15 次。deep-dive 提到这里用 fp16 做 inverse，因为矩阵是 16x16，且元素范围适合 fp16，最后再转成 bf16 写 workspace。

### 5.8 K1 为什么适合高 occupancy

K1 的 shared memory 生命周期被刻意复用：

- load 阶段需要 `q/k/g`。
- decay 后需要 `k_decayed/q_decayed/k_inv/L/INV/Mqk`。
- 两段生命周期不重叠，所以 `SharedStorageK1` 用 union 复用 shared memory。

再配合：

```cpp
__launch_bounds__(NumThreads, 8)
```

目标是让 K1 在 SM 上有足够多 CTA，弥补部分 register pressure。

## 6. K2: `_flash_kda_fwd_recurrence`

K2 的目标是沿 sequence 顺序消费 K1 的 chunk 包：

```text
load chunk package -> compute output -> update recurrent state -> next chunk
```

K2 launch 参数：

```cpp
dim3 grid_k2(N, H);
dim3 block_k2(192);
```

每个 CTA 处理一个 `(sequence, head)`。

### 6.1 warp specialization

192 threads 正好 6 个 warp：

| warp | 角色 |
|---:|---|
| 0-3 | MMA warp，做 GEMM 和 state update |
| 4 | LOAD warp，发 TMA load |
| 5 | STORE warp，发 TMA/manual store |

K2 使用：

```text
Input pipeline:  3 stages
Output pipeline: 2 stages
```

LOAD warp 提前读下一些 chunk 的 `v/beta/workspace`，MMA warp 消费当前 stage，STORE warp 写已经算完的 output。

### 6.2 state 初始化

K2 shared memory 里维护：

$$
S\in\mathbb{R}^{D\times D}
$$

代码名是 `state_acc`。初始化分三种：

1. 没有 `initial_state`: 清零。
2. bf16 state: TMA load 到 `state_acc`。
3. fp32 state: TMA load 到 fp32 临时 buffer，再转 bf16 到 `state_acc`。

因此主计算路径中，state 在 shared memory 里是 bf16 存储；MMA accumulation 是 fp32，然后写回 bf16 state。这是 deep-dive 中提到的精度/性能折中。

### 6.3 K2 每个 chunk 的矩阵语义

K2 读入当前 chunk：

$$
V,\quad K_\gamma,\quad Q_\gamma,\quad K_{\text{right}},\quad G,\quad INV,\quad Mqk
$$

第一步，旧 state 对当前 chunk 的贡献：

$$
U_0=K_\gamma S
$$

$$
O_0=Q_\gamma S
$$

代码中 `u_acc` 存 $U_0$，`out_acc` 存 $O_0$。

第二步，构造未解三角依赖前的 delta value：

$$
U_1=(V-U_0)\odot\sigma(\beta)
$$

第三步，用 K1 的 inverse 解 chunk 内依赖：

$$
U=INV\cdot U_1
$$

这一步就是简化版里的有效写入：

$$
U=D
$$

也就是说当前代码没有显式 materialize 简化版里的 $W=AK_\gamma$。它把：

$$
D=A(V-K_\gamma S_0)
$$

拆成：

$$
U_0=K_\gamma S_0
$$

$$
U_1=\operatorname{Diag}(\beta)(V-U_0)
$$

$$
D=INV\cdot U_1
$$

这里 $A$ 被拆成 $INV$ 和 $\operatorname{Diag}(\beta)$ 两部分，其中 $\operatorname{Diag}(\beta)$ 在 K2 中通过逐元素乘 `sigmoid(beta)` 完成。

第四步，输出：

$$
O=O_0+Mqk\cdot D
$$

这对应简化版：

$$
O=Q_\gamma S_0+ED
$$

其中 `Mqk` 就是三角化后的 $E$。

第五步，更新 state：

$$
S_{new}=\operatorname{Diag}(G)S_{old}+K_{\text{right}}^\top D
$$

逐行看就是：

$$
S_{new,d,:}=G_dS_{old,d,:}+\sum_{t=0}^{C-1}K_{\text{right},t,d}D_{t,:}
$$

代码中这一段在 Phase 6。它用 `k_restored_t` 做转置 layout，配合 `MOVM_T` 和 transposed C load/store，尽量让 $D$ 留在寄存器里，避免额外 shared memory 往返。

### 6.4 K2 代码 Phase 对照

`fwd_kernel2.cuh` 中 MMA 主体的注释可以这样对照：

| 代码 Phase | 数学含义 |
|---|---|
| Phase 1 | 同时计算 $K_\gamma S$ 和 $Q_\gamma S$ |
| Phase 2 | 把 output accumulator 转 bf16，加载 $V/INV/\beta$ |
| Phase 3 | 算 $(V-K_\gamma S)\odot\sigma(\beta)$，再乘 `INV` 得到 $D$ |
| Phase 4 | 算 `Mqk @ D` 并加到 output |
| Phase 5 | 把 output 写到 output staging shared memory |
| Phase 6 | 算 $\operatorname{Diag}(G)S+K_{\text{right}}^\top D$，更新 `state_acc` |

每个 MMA warp 负责 32 个 output columns，也就是两个 16x16 block。4 个 MMA warp 合起来覆盖 $D=128$ 的完整 value 维。

### 6.5 output 和 final_state store

STORE warp 写 output 时：

- 满 chunk 使用 TMA store。
- 尾 chunk 使用手写 loop，只写 `actual_len` 行，避免覆盖下一条 sequence。

final state：

- bf16 final state: 直接从 `state_acc` TMA store。
- fp32 final state: 先把 bf16 `state_acc` 转 fp32，再 TMA store。

## 7. 和简化版公式的完整对照

简化版主线是：

```text
Gamma = exp(cumsum(g_raw))
Q_gamma = Gamma * Q
K_gamma = Gamma * K
K_inv   = K / Gamma

B = K_gamma @ K_inv.T
A = inverse_lower(I + strict_lower(diag(beta) @ B)) @ diag(beta)

D = A @ (V - K_gamma @ S0)
O = Q_gamma @ S0 + tril(Q_gamma @ K_inv.T) @ D
S_C = diag(Gamma_C) @ S0 + K_right.T @ D
```

当前 kernel 的实际拆分是：

```text
K1:
  q_decayed   = scale * Q_gamma
  k_decayed   = K_gamma
  k_inv       = K / Gamma
  k_restored  = K_right
  L           = strict_lower(diag(sigmoid(beta)) @ (k_decayed @ k_inv.T))
  INV         = inverse(I + L)
  Mqk         = tril(q_decayed @ k_inv.T)
  g_total     = Gamma_C

K2:
  U0          = k_decayed @ S
  O0          = q_decayed @ S
  D_pre       = (V - U0) * sigmoid(beta)
  D           = INV @ D_pre
  O           = O0 + Mqk @ D
  S           = diag(g_total) @ S + k_restored.T @ D
```

这里有两个实现层面的变化：

1. 简化版的 $A$ 没有直接 materialize。kernel 只 materialize $INV$，把 $\operatorname{Diag}(\beta)$ 放到 K2 中逐元素乘。
2. `q_decayed` 已经乘了 `scale`，所以 K2 输出不再单独乘 attention scale。

## 8. deep-dive 中几个设计点如何落到代码

### 8.1 为什么 `CHUNK=16`

`CHUNK=16` 让三件事成立：

- $2^{cumsum(g)}$ 的范围更容易被 bf16 表示。
- $16\times16$ triangular inverse 可以用单 warp Neumann expansion。
- K1/K2 的小矩阵乘都能自然映射到 SM80 MMA atom。

如果 chunk 变大，`INV` 的构造成本和数值范围都会变复杂，K2 的 shared memory 和 register 压力也会上升。

### 8.2 为什么用 base-2 exponent

K1 把 gate scale 预先乘 $\log_2(e)$，后续所有指数都可以写成：

$$
2^x
$$

代码中对应：

```cpp
ex2_approx_ftz_f32(x)
```

这比直接 `expf` 更适合高吞吐 kernel。

### 8.3 为什么 state 在 shared memory 里用 bf16

K2 的 state 是 $128\times128$。如果用 fp32 存 shared memory，容量和带宽压力都会翻倍。当前实现：

```text
shared state: bf16
MMA accumulate: fp32
optional external state: bf16/fp32
```

这就是 deep-dive 里说的“state update 用 fp32 FMA，但中间 state 用 bf16 保存”。

### 8.4 为什么 K2 用寄存器转置

K2 的 $D$ 既要参与 output：

$$
Mqk\cdot D
$$

又要参与 state update：

$$
K_{\text{right}}^\top D
$$

如果每一步都把 $D$ 写回 shared memory，再换 layout 读出来，会多很多 shared memory 流量。代码使用 `MOVM_T` 做 register-file transpose，让 $D$ 在寄存器里转换成后续 MMA 需要的 fragment layout。

## 9. 看代码时的断点

建议按下面顺序看：

1. `flash_kda.cpp`: 看 `DISPATCH_STATE`，确认有哪些 `HasStateIn/HasStateOut/StateFP32/IsVarlen` 分支。
2. `fwd_launch.cu`: 看 workspace 六段 pointer 切分。
3. `fwd_kernel1.cuh`: 看 `TMA load -> q/k norm -> gate cumsum -> decay_apply -> L/Mqk -> INV -> TMA store`。
4. `utils.cuh`: 看 `neumann_inv_fused_1warp` 如何实现 $(I+L)^{-1}$。
5. `fwd_kernel2.cuh`: 看 Phase 1-6，把每个 phase 对到第 6 节的矩阵公式。

如果只想快速确认实现是否符合算法，重点盯这几组变量：

| 算法变量 | K1/K2 变量 |
|---|---|
| $Q_\gamma$ | `q_decayed` |
| $K_\gamma$ | `k_decayed` |
| $K_{\gamma^{-1}}$ | `k_inv` |
| $K_{\text{right}}$ | `k_restored` |
| $G$ | `g_total` |
| $(I+L)^{-1}$ | `INV` |
| $E$ | `Mqk` |
| $D$ | K2 中 Phase 3 后的 `u_bf16/tCrB_u_arr` |
| $S$ | `state_acc` |

## 10. 常见误读

### 10.1 `g_total` 不是 log

K1 中 `g_total` 先短暂保存 prefix sum 的最后一项，之后立刻被改写成：

$$
g\_total_d=2^{c_{C-1,d}}
$$

所以 K2 用它直接乘 state。

### 10.2 `beta` 在 kernel 内 sigmoid

API 传进来的 `beta` 是 logits。K1 构造 `L` 和 K2 构造 `D_pre` 时都会使用：

$$
\sigma(\beta)
$$

外部不能提前 sigmoid，否则语义会错。

### 10.3 `INV` 不包含右乘的 `Diag(beta)`

简化版里的：

$$
A=(I+L)^{-1}\operatorname{Diag}(\beta)
$$

当前 kernel 只存：

$$
INV=(I+L)^{-1}
$$

而 $\operatorname{Diag}(\beta)$ 在 K2 中通过：

$$
(V-K_\gamma S)\odot\sigma(\beta)
$$

完成。

### 10.4 `Mqk` 已经是 causal 下三角

K2 做：

$$
O=O_0+Mqk\cdot D
$$

时不再额外 mask，因为 K1 已经把 `Mqk` 的上三角清零。

### 10.5 varlen 的 tile offset 是 kernel 内线性扫描

当前代码没有传预计算的 `tile_offsets`。K1/K2 都会根据 `cu_seqlens` 线性扫描计算 tile base。sequence 数很大时，这是一个可以关注的开销点。

## 11. 一句话收束

从算法看，FlashKDA forward 做的是：

$$
D=(I+\operatorname{StrictTril}(\operatorname{Diag}(\beta)B))^{-1}\operatorname{Diag}(\beta)(V-K_\gamma S)
$$

$$
O=Q_\gamma S+ED
$$

$$
S_{new}=\operatorname{Diag}(G)S+K_{\text{right}}^\top D
$$

从实现看，K1 负责提前构造 $K_\gamma,Q_\gamma,K_{\text{right}},G,INV,E$，K2 负责把这些 chunk 包串起来做 recurrence。
