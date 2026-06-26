# FlashKDA 优化计划

这份文档记录当前 forward kernel 的问题判断和后续优化路线。目标不是一次性重写，而是按性价比逐步推进：先解决 K2 CTA 数不足，再考虑 decode 专用路径，最后再评估 CP / scan 级别的结构改造。

## Roadmap

- [ ] Profile baseline and decide the first optimization target.
  - [ ] Measure K1/K2 time split.
  - [ ] Check whether K1 prologue is a significant fraction of forward time.
  - [ ] Check whether K2 is mainly limited by low CTA count.

**K1 Prologue**

- [ ] Optimize Q/K L2 norm shared-memory access.
  - [ ] Replace scalar BF16 access with `BF16x8` packed 16B access.
  - [ ] Target SASS: `LDS.128` / `STS.128`.
  - [ ] Avoid keeping `q_vals[8]` / `k_vals[8]` as long-lived FP32 arrays.

- [ ] Split the initial TMA mbarrier.
  - [ ] Use `qk_barrier` for `q` and `k`.
  - [ ] Use `aux_barrier` for `beta`, `g_bf16`, and `dt_bias`.
  - [ ] Start Q/K L2 norm as soon as q/k TMA completes.

**K2 / Prefill**

- [ ] Split K2 by value/state dimension.
  - [ ] First try `V_BLOCK = 64`.
  - [ ] Change `grid_k2` to `N x H x V_blocks`.
  - [ ] Evaluate `V_BLOCK = 32` only if `V_BLOCK = 64` is not enough.

**Decode**

- [ ] Add a decode-specific kernel.
  - [ ] Use `grid_decode = B x H x V_blocks`.
  - [ ] Avoid K1 workspace.
  - [ ] Prioritize V-dimension parallelism.

**Architecture Experiments**

- [ ] Prototype WGMMA / UMMA backend after K2 V split.
  - [ ] Keep the math path unchanged.
  - [ ] Only replace the MMA backend.

- [ ] Prototype superchunk CP / scan as a longer-term experiment.
  - [ ] Start with segment-level scan.
  - [ ] Treat full chunk-level scan as a later research path.

Current decision rule:

- If K1 prologue is significant, optimize K1 prologue first.
- Otherwise, start with K2 `V_BLOCK = 64`.

## 1. 当前判断

当前 forward 分为两个 kernel：

```text
K1: chunk-local prepare
K2: chunk recurrence and output
```

K1 的并行度来自 chunk：

```text
grid_k1 = total_tiles x H
```

非 varlen 时：

```text
total_tiles = B * ceil(T / CHUNK)
```

所以 K1 通常 CTA 很多。

K2 的并行度只来自 batch 和 head：

```text
grid_k2 = N x H
```

非 varlen 时 $N=B$。因此在 batch=1 或 TP 后每卡 head 数较少时，K2 很容易 CTA 不够。K2 内部沿 chunk 顺序更新 state：

```text
for chunk in chunks:
    update state
    write output
```

中间 chunk 的 state 只在 shared memory 中滚动，最后才写 `final_state`。这省掉了中间 state 的 global memory 写回，但没有降低时间依赖深度。

当前 FlashKDA v1 还需要明确几件事：

```text
CHUNK = 16
t_tiles = ceil(seq_len / 16)
K2 是一个 CTA 顺序扫 t_tiles
CTA 内没有 superchunk 聚合
没有 chunk transform scan
没有 cluster-level state 聚合
```

虽然源码包含 `cluster_sm90.hpp` / `cluster_launch.hpp`，并且 TMA barrier 使用了 cluster transaction barrier 类型，但实际 launch 仍是普通 CUDA launch：

```cpp
kernel<<<grid, block, smem, stream>>>
```

没有使用 cluster launch，也没有跨 CTA 聚合 state。

当前主计算也不是 Hopper WGMMA，而是 SM80 warp-level MMA：

```cpp
MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>
```

这个选择兼容性好，Hopper / Blackwell / Ada 都能走，但没有吃到 Hopper WGMMA 或 Blackwell UMMA 的更大 tile 能力。

## 2. cuLA 对照

本地 `/home/tangpanyu/reps/cuLA` 里，KDA 的 chunk 配置和 FlashKDA 不同：

```text
cuLA Hopper / H200 fused prefill:     chunk_size = 64
cuLA Blackwell / B200/GB200 prefill:  chunk_size = 64
cuLA default chunk_kda:               chunk_size = 64
```

对应代码：

```text
cula/kda/hopper_fused_fwd.py      chunk_size = 64
cula/kda/blackwell_fused_fwd.py   chunk_size = 64
cula/kda/chunk.py                 chunk_size = 64
```

但是 cuLA 的 64-token chunk 内部仍然使用 16-token subchunk。文档 `docs/cuLA kda讲解.md` 中记录了 SM100 forward intra 的结构：

```cpp
static constexpr int SubTileT = 16;
static constexpr int TileT = 64;
```

所以准确说法是：

```text
FlashKDA: 外层 CHUNK = 16
cuLA:     外层 TileT = 64，内部 SubTileT = 16
```

这说明 `16` 更适合作为局部 solve / inverse 的粒度，而不一定适合作为整个 recurrent chunk 的外层粒度。cuLA 仍然选择外层 64，再在内部用 16 subchunk 控制数值范围和三角求逆复杂度。

## 3. 主要瓶颈

K2 的时间依赖来自 recurrent state：

$$
S_{i+1}=F_i(S_i)
$$

当前实现让一个 CTA 负责一个 `(sequence, head)`，顺序扫完所有 chunks。因此：

```text
B=1, H=96  -> 96 CTA
B=8, H=96  -> 768 CTA
```

提高 batch 后利用率上升，说明 K2 的 CTA 数不足是主要问题之一。

同时需要注意：如果 profile 显示 K1 prologue 占比不小，那么 K1 prologue 的局部优化应提前到 K2 V split 之前验证。原因是这部分不改变 KDA 主数学结构，风险低于重构 K2，但当前已经有两个明确机会：

```text
Q/K L2 norm shared access 未充分 vectorize / 可能有 bank conflict
初始 TMA load 使用单个 mbarrier，依赖粒度过粗
```

因此当前优先级不是固定的“永远先 K2”。更合理的判断是：

```text
K1 prologue 明显占时 -> 先做 K1 prologue local optimization
K1 占比不高、K2 利用率低 -> 先做 K2 V split
```

### 3.1 K1 局部问题：Q/K L2 norm shared memory bank conflict

K1 中 Q/K L2 normalization 当前每个线程读取连续 8 个 BF16：

```cpp
constexpr int ELEMS_PER_THREAD = 8;
int my_col = (threadIdx.x % THREADS_PER_ROW) * ELEMS_PER_THREAD;

for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
    float qv = bf16_to_f32(q_smem[my_row * D + my_col + i]);
    float kv = bf16_to_f32(k_smem[my_row * D + my_col + i]);
    ...
}
```

这个访问本身适合 16B vector load/store，因为 8 个 BF16 正好是 16 bytes。但是标量 BF16 写法不一定生成 `LDS.128`。局部实验显示，编译器可能把 8 个 BF16 load 合并成：

```text
LDS.64 x 2
```

而不是：

```text
LDS.128 x 1
```

在 `base = tid * 16B` 的访问模式下，每条 `LDS.64` 理想需要 2 个 wavefront，实际可能是 4 个 wavefront，因此两条 `LDS.64` 会产生：

```text
actual wavefronts = 8
ideal wavefronts = 4
bank conflicts = 4
```

Q 和 K 都执行该模式，写回也可能有类似问题。这块不是全局主瓶颈的确定结论，但它是 K1 中明确存在的局部低效点。

建议实验方向：

```text
使用 alignas(16) 的 BF16x8 packed 类型
让每个线程以 16B 粒度读取/写回自己的 8 个 BF16
目标 SASS: LDS.128 / STS.128
避免 q_vals[8] / k_vals[8] 长期保持 FP32，降低寄存器压力
```

推荐结构：

```cpp
struct alignas(16) Bf16x8 {
    BF16 x[8];
};

Bf16x8 q_pack = q_vec[vec_idx];
Bf16x8 k_pack = k_vec[vec_idx];
```

计算平方和时临时转 FP32。归一化写回时从 packed BF16 再转一次 FP32，而不是保存 `q_vals[8]` / `k_vals[8]`：

```text
packed BF16 常驻寄存器
FP32 只作为临时变量
```

该改法的目标不是改变数学路径，而是减少 shared memory bank conflict、减少 shared load/store 指令数，并降低 FP32 数组寄存器占用。

### 3.2 K1 局部问题：TMA mbarrier 粒度过粗

K1 当前把多个独立 TMA load 挂在同一个 `tma_load_barrier` 上：

```text
q
k
beta
g_bf16
dt_bias
```

随后在 Q/K L2 norm 前统一等待：

```cpp
shared_storage.tma_load_barrier.wait(0);
```

这会形成过粗的依赖：Q/K L2 norm 只依赖 `q` 和 `k`，但必须等 `beta`、`g_bf16`、`dt_bias` 也全部完成。更合理的拆分是至少使用两组 barrier：

```text
qk_barrier:
    q
    k

aux_barrier:
    beta
    g_bf16
    dt_bias
```

目标调度：

```text
issue q/k TMA
issue beta/g/dt_bias TMA
compute a_log_exp
wait qk_barrier
run Q/K L2 norm
wait aux_barrier
run gate + cumsum and later beta path
```

这样可以让 Q/K L2 norm 与 auxiliary TMA 的尾部重叠，降低全 CTA 等待粒度。需要注意 `arrive_and_expect_tx` 的 byte count 必须按 barrier 分组精确计算。

这项优化应通过 profile 验证收益。如果 `q/k` TMA 本身已经是最长路径，拆 barrier 的收益可能有限；如果 `g_bf16` 或小张量 TMA 的尾部拖住了 Q/K norm 开始时间，收益会更明显。

## 4. 优先级一：K2 按 V/state 切分

K2 中 state 的形状是：

$$
S\in\mathbb{R}^{K\times V}
$$

对固定的 `q/k/g/beta`，不同 value column 之间基本独立。可以把 K2 改成按 `V` 维切：

```text
grid_k2 = N x H x V_blocks
```

每个 CTA 只负责：

```text
S[:, v0:v1]
v[:, v0:v1]
out[:, v0:v1]
```

例如：

```text
V = 128
V_BLOCK = 64 -> 2 CTA per (N,H)
V_BLOCK = 32 -> 4 CTA per (N,H)
V_BLOCK = 16 -> 8 CTA per (N,H)
```

### 4.1 可以按 V 切的部分

这些张量或计算带 value 维，适合放进 V-block CTA：

```text
v_tile
out_tile
state_acc[:, v0:v1]
old_v = S[:, v0:v1]^T @ k
out = S[:, v0:v1]^T @ q
U / v_delta[:, v0:v1]
INV @ U[:, v0:v1]
Mqk @ U[:, v0:v1]
state update for S[:, v0:v1]
```

### 4.2 不应重复计算的部分

这些是 K1 产出的 chunk-local 信息，没有 V 维，所有 V-block 共享：

```text
q_decayed
k_decayed
k_restored
g_total
INV
Mqk
beta
```

V-block 版本的 K2 会重复读取这些共享 workspace。这个会增加带宽压力，但可以换来更多 CTA 和更好的 latency hiding。需要用 profile 判断 `V_BLOCK=64` 和 `V_BLOCK=32` 哪个更合适。

### 4.3 第一版建议

第一版优先尝试：

```text
V_BLOCK = 64
```

原因：

```text
CTA 数翻倍
state_acc shared memory 减半
代码改动小于 V_BLOCK=32
共享 workspace 重复读取只增加 2 倍
```

如果 `V_BLOCK=64` 提升不够，再尝试 `V_BLOCK=32`。

## 5. 优先级二：decode 专用 kernel

Decode 时通常 $T=1$，没有 chunk 维可切。单步公式为：

$$
\bar S=\operatorname{Diag}(\exp(g))S
$$

$$
v_\Delta=\beta(v-\bar S^Tk)
$$

$$
S_{\text{new}}=\bar S+kv_\Delta^T
$$

$$
o=S_{\text{new}}^Tq
$$

Decode 最直接的并行维度仍然是 `V`：

```text
grid_decode = B x H x V_blocks
```

每个 CTA 处理：

```text
S[:, v0:v1]
v[v0:v1]
o[v0:v1]
```

这个路径不需要 K1，也不需要 `INV/Mqk`。它应该独立于 prefill kernel 设计。

如果 `B * H * V_blocks` 仍然不够，只能进一步考虑：

```text
K split + reduction
continuous batching
grouped decode / persistent kernel
```

其中 `K split` 会引入跨 CTA reduction，复杂度明显高于 V split。

## 6. 优先级三：WGMMA / UMMA 升级

当前 kernel 使用 SM80 warp-level MMA。升级到 WGMMA / UMMA 是合理方向，但不应该排在 V split 前面。

原因：

```text
当前主要症状是 K2 CTA 数不足
低 CTA 数时，单 CTA 更强不一定能提高整体 SM 利用率
WGMMA 往往会增加单 CTA 资源需求，可能进一步降低 resident CTA
```

因此 WGMMA 更适合在下面条件满足后再做：

```text
K2 已经按 V split 增加 CTA 数
profile 显示 tensor core compute 成为瓶颈
TMA / memory / barrier stall 不再是主因
```

短期建议保留两条路径：

```text
SM80 MMA path: 兼容、低风险、便于 correctness 对齐
WGMMA/UMMA path: 面向 Hopper/Blackwell 的性能实验分支
```

低占用场景下，SM80 MMA 未必比 WGMMA 差。因为瓶颈可能不是单 CTA 算力，而是：

```text
CTA wave 数不足
TMA 等待
pipeline 同步
state recurrence 串行依赖
```

所以 WGMMA 的优先级应低于 `V/state split`，但高于完整 CP/scan。它是中期性能分支，不是第一步结构修复。

## 7. 优先级四：CP / scan 方案

如果要在 prefill 中把 context/chunk 维也并行化，需要把每个 chunk 或 segment 表示成 affine transform：

$$
S_{\text{out}}=MS_{\text{in}}+H
$$

两个相邻 transform 可以组合：

$$
(M_2,H_2)\circ(M_1,H_1)=(M_2M_1,\;M_2H_1+H_2)
$$

这个组合满足结合律，所以可以做 reduction 或 scan。

### 7.1 只需要 final_state

如果只需要最后的 state，可以做 tree reduction：

```text
level 0: chunk transforms
level 1: pairwise combine
level 2: pairwise combine
...
```

依赖深度从 $O(N)$ 降为 $O(\log N)$，其中 $N$ 是 chunk 数。

### 7.2 需要所有 token output

推理 prefill 通常还需要每个 token 的 output。每个 chunk 的 output 需要该 chunk 的输入 state，因此只做 final reduction 不够，必须做 prefix scan，或者保存 segment 边界后在 segment 内重算。

可选方案：

```text
Blelloch scan:
    work efficient, depth O(log N), operator 代价高时更合理

superchunk scan:
    每个 segment 内顺序扫若干 chunks
    对 segment transform 做 scan
    segment 内拿到 prefix state 后再生成 output
```

### 7.3 为什么不是短期优先项

KDA 的 transform 很大：

$$
M\in\mathbb{R}^{128\times128}
$$

$$
H\in\mathbb{R}^{128\times128}
$$

combine 需要：

```text
M2 @ M1
M2 @ H1 + H2
```

这会引入大量 workspace、global memory traffic 和新的 rounding 点。除非 V split 和 batching 都不能解决目标场景，否则不建议先做完整 CP/scan。

### 7.4 cluster 聚合

Cluster-level state 聚合可以作为 CP/scan 的实现细节之一，但当前 FlashKDA 没有做。

如果未来做 cluster 版本，需要先明确聚合对象：

```text
不是直接聚合最终 state
而是聚合 segment/chunk 的 affine transform (M,H)
```

否则后段 chunk 无法得到正确的入口 state。cluster 只解决通信和调度方式，不改变数学依赖。

## 8. 数值风险

当前图中 `flash_kda` 相对 fp64 gold 的误差通常高于 FLA `chunk_kda`。主要嫌疑不是 `CHUNK=16` 本身，而是低精度中间路径：

```text
state_acc in bf16
workspace q/k/k_restored/INV/Mqk in bf16
INV computed through fp16 path
approx tanh / ex2
更多 chunk 边界量化点
```

后续优化需要同时记录：

```text
output error vs fp64 gold
final_state error vs fp64 gold
exact match vs torch_ref when applicable
```

如果改 K2 V split，理论上数值路径可以保持接近现有 K2，误差风险较小。CP/scan 会改变组合顺序，误差风险更大。

外层 chunk size 也需要重新评估。cuLA 的事实是：

```text
外层 chunk_size = 64
内部 subchunk = 16
```

这说明 64-token chunk 并没有想象中必然带来不可接受误差。FlashKDA 现在误差更大，更可能来自 bf16 state、bf16 workspace、fp16 inverse 和近似指令组合，而不是 `CHUNK=16` 没有发挥作用。

## 9. 建议里程碑

### M1: Profile baseline

目标：

```text
确认 K1/K2 时间占比
确认 K2 SM active / occupancy / stall reason
确认 B/H/V 对利用率的影响
```

推荐 case：

```text
B=1, H=96, T=8192
B=8, H=96, T=8192
B=1, H=1,  T=8192
```

额外记录 K1 局部指标：

```text
Q/K L2 norm shared load/store 的 LDS/STS 形态
shared memory bank conflict load/store
TMA wait 前后的 stall
q/k TMA 与 beta/g/dt_bias TMA 是否存在可重叠尾部
```

### M1.5: K1 local optimization experiments

目标：

```text
Q/K L2 norm 改成 BF16x8 packed 访问
目标 SASS: LDS.128 / STS.128
移除 q_vals[8] / k_vals[8] 长期 FP32 常驻
```

同步实验：

```text
把 K1 初始 TMA barrier 拆成 qk_barrier 和 aux_barrier
q/k 完成后先开始 Q/K L2 norm
aux barrier 只保护 beta/g_bf16/dt_bias 后续使用
```

验证：

```text
bank conflict 是否下降
register count / occupancy 是否变化
K1 wall time 是否下降
整体 forward 是否受益
数值误差不显著变化
```

### M2: K2 V_BLOCK=64

目标：

```text
grid_k2 = N x H x 2
state_acc 改为 [K,64]
out/v TMA 或手写 store/load 支持 V slice
final_state 按 V slice 写回
```

验证：

```text
test_fwd_full small cases
test_fwd fixed and varlen
plot.png / plot_varlen.png 误差不显著变坏
```

### M3: K2 V_BLOCK=32

目标：

```text
grid_k2 = N x H x 4
评估 CTA 增加和 workspace 重复读取之间的 tradeoff
```

### M4: Decode V split

目标：

```text
独立 decode kernel
grid = B x H x V_blocks
不经过 K1 workspace
```

### M5: WGMMA/UMMA prototype

目标：

```text
保持现有数学路径不变
只替换主 GEMM tile 的 MMA backend
对比 SM80 MMA vs WGMMA/UMMA
```

前置条件：

```text
先完成 K2 V split
确认 profile 中 compute 占比足够高
```

不建议在 CTA 数不足的 baseline 上直接做 WGMMA，因为容易把结构性低并行度误判成 MMA backend 问题。

### M6: Superchunk CP prototype

目标：

```text
只做实验原型
segment 内顺序扫
segment 间 affine transform scan
评估 workspace 和数值误差
```

### M7: Cluster aggregation experiment

目标：

```text
只作为 CP/scan 的通信实现实验
验证 cluster 内聚合 (M,H) 的成本
不作为第一阶段性能主线
```

## 10. 当前推荐路线

短期路线：

```text
Profile 确认 K1 prologue 与 K2 的相对占比
如果 K1 prologue 占比不小，先做 K1 prologue local optimization:
    BF16x8 packed Q/K L2 norm
    qk_barrier / aux_barrier 拆分
K2 先做 V split
Decode 单独做 V split
```

中期路线：

```text
WGMMA/UMMA experimental backend
重新评估外层 chunk_size = 64 + 内部 subchunk = 16
```

中长期路线：

```text
superchunk scan / CP
cluster aggregation for segment transforms
K split + reduction for extreme low B/H decode
```

不要一开始就做完整 chunk-level scan。它理论上漂亮，但工程成本和数值风险都高。当前最直接的问题是 CTA 数不足，V/state split 是性价比最高的第一步。
