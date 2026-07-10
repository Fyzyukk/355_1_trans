# MXFP8 GEMM Pipeline 结构说明

针对当前 `gemm_a8w8_mxfp8_kernel_template.hpp` 的流水实现。目前仅支持 **K=256**
（两个 K-tile，B_K=128），因此没有主循环，pipeline 完全展开为 prologue + 两个块。

## Tile / 寄存器模型

- 输出 block：`B_M×B_N = 256×256`，拆成 **2×2 = 4 个 128×128 子tile** → `v_c[hm][hn]`
  - `hm ∈ {0,1}`：M 上/下半区（HALF_B_M=128）
  - `hn ∈ {0,1}`：N 左/右半区（HALF_B_N=128）
- K 方向：`K=256` = 2 个 K-tile（B_K=128），用 `stage ∈ {0,1}` 表示，LDS 双缓冲（`smem_*4`）。

### 寄存器分配（单 v_a、双 v_b）

```cpp
vtype_a v_a;            // 单个：一次持有一个 (k, hm) 的 A 片，复用
vtype_b v_b[2];         // 两个：同一 (k) 下的两个 half_n
vtype_c v_c[2][2];      // 4 个子tile累加器（跨整个 K 累加）
vtype_c v_mma[2];       // MMA 原始结果，软件流水错开 mma 与 scale
vtype_sfa v_sfa[2][2];  // [half_tile_m][tile_k] A scale, 每元素 E_M*E_K 字节
vtype_sfb v_sfb[2][2];  // [half_tile_n][tile_k] B scale, 每元素 E_N*E_K 字节
```

### offset 约定

```cpp
ga_offset(half_tile_m, tile_k)   // global A: M 半区 + K tile
gb_offset(half_tile_n, tile_k)   // global B
sa_offset(stage, half_tile_m)    // LDS A: stage 双缓冲 * 2 + half
sb_offset(stage, half_tile_n)    // LDS B
sfa_offset(half_tile_m, tile_k)  // global SFA: (HALF_B_M/GROUP_M)*stride + tile_k*NUM_KGROUPS
sfb_offset(half_tile_n, tile_k)  // global SFB: 对称
```

## 缩放（软件路径）

每个子tile的 partial 由 `scale_c_group` 逐元素后处理：

```
acc[i] += c_mma[i] * e8m0_to_f32(scale_a[a_idx]) * e8m0_to_f32(scale_b[b_idx])
i     = m_rep*(E_N*ELEM_C) + n_rep*ELEM_C + pack
a_idx = m_rep*ELEM_C + pack      // 8 个 A scale (E_M*ELEM_C)
b_idx = n_rep*ELEM_C + pack      // 16 个 B scale (E_N*ELEM_C)
```

`pack`（C fragment 组内索引）同时选中 A、B scale 的组内偏移。E_K==ELEM_C==4。

## 执行流程

### Prologue
1. 载入 stage0 两个 half 的 sfa/sfb + async_load a/b（half0, half1）
2. `wave_id_n==1` barrier；等 vmcnt；barrier
3. 载入 stage1 两个 half 的 sfa/sfb + async_load a/b
4. 等 vmcnt；barrier
5. 预读 `v_a=ra(stage0,hm0)`、`v_b[0]=rb(stage0,hn0)`；async_load 剩余 b
6. `v_mma[0] = mma(v_a, v_b[0])`  ——第一个 (m0,n0) partial

### 块1（第一个 K-tile 的计算 + 第二 K-tile 预取交织）
- 读 `v_b[1]`，`v_mma[1]=mma(v_a,v_b[1])`
- `scale_c_group` 把 v_mma[0] 累加进 v_c[0][0]
- 换 `v_a`，继续 mma/scale，覆盖 v_c[0][1]、v_c[1][0]
- 每次 scale 后用 inline asm pin 住 v_c + `sched_barrier_pairs<8,4,0>` 控制调度

### 块2（第二个 K-tile 收尾）
- 读 stage1 的 v_a/v_b，做剩余 mma/scale
- 覆盖 v_c[1][1]、以及各子tile的第二次 K 累加
- 最后一批 scale 收尾

### Epilogue / store
- `wave_id_n==0` barrier
- `partition_layout_c<VEC_C>` 建 C 输出布局
- `store<VEC_C>` 写 4 个子tile到 global C

## 调度控制

- `s_setprio(1/0)`：抬高/恢复 MMA 段优先级
- `sched_barrier_pairs<Pairs,VALU_CNT,Group>`：交替发 MFMA / VALU，隐藏 scale 的 VALU 延迟
- inline asm `"+v"(v_c_pin)`：阻止编译器过早移动 v_c 的读写

## 限制 / TODO

- 仅 K=256（无主循环）。支持任意 K 需补 `for (tile; tile<loops-2; tile+=2)` 主循环。
- MMA 为普通 fp8 MMA + 软件 scale。硬件 scaled-MMA 方案见 `SCALED_MMA_TODO.md`。
- scale 的物理排布（c_mma 元素序、sfa/sfb 字节序）需 `-v` 数值校验确认。
