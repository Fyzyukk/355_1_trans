# MXFP8 GEMM 调试进度 (更新至 2026-07-10)

> 本文记录**当前进度、已钉死的结论、以及下一步怎么做**。下次直接读本文即可接手。
> 通用方法论见 `LAYOUT_VERIFY_METHODOLOGY.md`;单指令真机 probe 见
> `../mxfp8_wmma_scale_probe/MMA_LAYOUT_PROBING_METHODOLOGY.md`。
>
> ⚠️ **重要更正**:本文早先版本(以及 `LAYOUT_VERIFY_METHODOLOGY.md` §4 坑3、
> `../mxfp8_wmma_scale_probe/` 的 §5.2 手推 C 表)对 16x16x32 swap_ab 的 C 布局
> **方向判断反了**。已用端到端实测探针 `probe_c_mma.cu` 钉死正确布局,见 §1。

---

## 0. 一句话现状

- **两个独立 bug**:①C 布局方向反了(**已修+已验证**);②tiled mma 把 4 个 K-group
  全加进一个 fragment 导致无法按 group 施加 scale(**已定方案,未落地**)。
- **下一步**:用 `step_k` 逐 K-group 计算 + 逐 group 乘 scale(方案A),并重写 `sfb`
  载入。详见 §3。

---

## 1. 已钉死的权威结论:16x16x32 swap_ab 的真实 C 布局(C-layout 0)

用 `probe_c_mma.cu`(端到端实测:构造唯一指纹 `C[m][n]=m*16+n`,喂真正的
`make_tiled_mma(...mfma_adaptor_swap_ab)` 读回)钉死:

```
单条 16x16x32:  每 lane 4 个累加寄存器 pk=0..3
  m = lane % W_M              (= lane % 16, = row)
  n = (lane / W_M) * 4 + pk   (= gk*4 + pk, gk=lane/16)  ← 每 lane 连续 4 列
```

推广到 block-tile(fragment `i = m_rep*(E_N*4) + n_rep*4 + pk`,与 `v_c`/scale 顺序一致):
```
m = m_rep*(W_M*T_M) + wave_id_m*W_M + row          row = lane % W_M
n = n_rep*(W_N*T_N) + wave_id_n*W_N + gk*4 + pk     gk  = lane / W_M
```

> 这**推翻**了早先"连续 4 行"的判断。早先错因:①手推预测表方向反;
> ②`probe_c_store.cu` 只跑 wave0,双射只覆盖 2048/16384 是"wave0 只占 1/8"的假象,
> 不是布局坏。现 `probe_c_store.cu` 已改跑全 8 wave。

**关键推论(scale 用):MFMA 在输入↔输出之间转置了 lane↔数据。**
- A 侧:输入行 = 输出 m = `lane%16`,**没转置** → `sfa`/`ra` 的 n/m 来源一致。
- B 侧:输入行(rb 用 `lane%W_N`)≠ 输出列(C 用 `(lane/W_M)*4+pk`),**被转置** →
  `sfb` **不能照抄 rb**,必须对齐 C-layout 0 的 n。

---

## 2. 已完成并验证(不用再动)

| 层 | 手写函数 | 验证探针 | 结论 |
|---|---|---|---|
| A global→LDS→reg | `ga`/`sa`/`ra` | `probe_ra.exe` | ✅ 0 mismatch, LDS 全覆盖 |
| B global→LDS→reg | `gb`/`sb`/`rb` | `probe_rb.exe` | ✅ 0 mismatch, LDS 全覆盖 |
| **C reg→global** | **`make_layout_gc`(新手写)** | `probe_c.exe` | ✅ 双射 16384/16384 + fragment 序 0 mismatch |
| C 布局 ground truth | — | `probe_c_mma.exe` | ✅ 实测 = C-layout 0 |

- **`make_layout_gc`**(kernel_template.hpp):已按 C-layout 0 手写,pk 连续列 →
  `VEC_C=4` 存储,epilogue 已接上。**这个是对的,别改。**
- **`make_layout_sfa`**:**碰巧对**(m=lane%W_M 与 C-layout 0 一致),不用动。
- 探针编译命令:
  ```sh
  export OPUS_INCLUDE_DIR=/root/workspace/aiter/csrc/include
  /opt/rocm/bin/hipcc probe_c_store.cu -I. -I$OPUS_INCLUDE_DIR -std=c++17 -O2 --offload-arch=gfx950 -o build/probe_c.exe
  # probe_c_mma.cu / probe_ra_roundtrip.cu / probe_rb_roundtrip.cu 同理
  ```

---

## 3. 下一步:方案A + step_k(未落地,按此执行)

### 3.1 根因(Bug2)

`make_tiled_mma` 的 `mma(v_a,v_b)` 内部 loop `EXPAND_K=E_K=4`(opus.hpp:2987),把
**4 个 32-K group 全累加进同一个 `v_c` fragment**。但 MXFP8 要求每个 K-group 先乘
各自的 `scale_a·scale_b` 再累加:
```
C[m][n] = Σ_kg (scale_a[m][kg]·scale_b[n][kg]) · (Σ_{k∈kg} A[m][k]·B[n][k])
```
所以事后的 `scale_c_group` 数学上不可能对。**这也是 unit-scale 也失败的原因**
(fragment 索引在"4 group 已混"的前提下全错)。

### 3.2 用 step_k 拆 K-group(已确认签名)

`opus.hpp:3070`:
```cpp
// 只算第 STEP_K 个 K-group 的 E_M*E_N 次 sub-mma,c 作为累加基,返回完整 vtype_c
auto v_part = mma.step_k(opus::number<kg>{}, v_a, v_b, zero_c);
```
- 返回值 = 完整 `vtype_c`(E_M*E_N*elem_c),但只累加了第 kg 个 group 的贡献。
- **不动 wave 分工**:8 wave 自动并行,和 mma() 一样。
- **保留**已验证的 v_a/v_b/v_c 尺寸与 A/B/C layout。

**前置任务(务必先做,task #5)**:写最小探针验证
`Σ_kg step_k<kg>(v_a,v_b,0)` 数值上 == `mma(v_a,v_b)`,钉死语义再动主流水。

### 3.3 主流水改造

把当前每处 `v_mma = mma(v_a, v_b)` + `scale_c_group(...)` 换成:
```cpp
// 对一个 128x128 子tile(一个 v_c[hm][hn]),一个 K-tile:
for (int kg = 0; kg < E_K; ++kg) {
    auto v_part = mma.step_k(number<kg>{}, v_a, v_b, zero_c);   // 只算 kg
    scale_and_accumulate(v_part, sfa, sfb, kg, v_c[hm][hn]);    // 乘 group scale 再累加
}
```
(注意保留现有的 sched_barrier/s_setprio/pin 调度控制,逐段迁移;当前流水是
prologue+2 块全展开、仅支持 K=256,见 `PIPELINE.md`。)

### 3.4 scale 乘法(方案A,按 C-layout 0 索引)

fragment `i = m_rep*(E_N*4) + n_rep*4 + pk`:
```
scale_a: 随 m_rep 变,4 个 pk 共享同一个 → 每 group 取 E_M 个,idx = m_rep*E_K + kg
scale_b: 随 n_rep 和 pk 变(每列不同)   → 每 group 取 E_N*4 个,idx = (n_rep*4+pk)*E_K + kg
acc[i] += v_part[i] * e8m0(scale_a[..]) * e8m0(scale_b[..])
```

### 3.5 重写 make_layout_sfb(task #6)

当前 sfb 照抄了 rb(n 来源 `lane%W_N`),但 C 输出 n=`(lane/W_M)*4+pk`。改:
```
n = n_rep*(W_N*T_N)   [E_N, y]
  + wave_id_n*W_N     [T_N, p = wave_id_n]
  + gk*4              [gk = W_N/... = 4, p = lane / W_M]   ← 关键:用 lane/W_M,不是 lane%W_N
  + pk                [pk = 4, y]                          ← 关键:补这个 y 维
```
每 lane 载 `E_N*4*E_K` 个 scale_b(现在只有 E_N*E_K,少了 pk=4 倍)。
**sfa 不动。** 改完写个探针验证 sfb 每 lane 的 scale 与 C-layout 0 的 n 对齐。

### 3.6 验证顺序(务必按此)

```sh
export OPUS_INCLUDE_DIR=/root/workspace/aiter/csrc/include
./build/probe_stepk.exe    # (task#5) Σ step_k == mma
make -j
# 兜底:unit-scale 下 scale=1,应等于普通 GEMM。这一关先过,说明 step_k+流水接线对。
MXFP8_UNIT_SCALE=1 ./build/gemm_a8w8_mxfp8.exe -v 1 -m 256 -n 256 -k 256 -b 1   # 期望 VALID
# 再 random scale 全绿(此时 sfb 必须已修对)
./build/gemm_a8w8_mxfp8.exe -v 1                                                # 期望 ALL VALID
```

---

## 4. Traits 速查(gemm_a8w8_mxfp8_traits<>, 默认)

```
B_M=B_N=256, B_K=128, HALF_B_M=HALF_B_N=128
W_M=W_N=16, W_K=32=GROUP_K   (一条 16x16x32 = 一个 K-group)
T_M=4, T_N=2, T_K=1  (8 wave = 4x2)
E_M=2, E_N=4, E_K=4  (E_K = B_K/W_K = 4 个 K-group,就是被全加的那 4 个)
VEC_C=4  (pk 连续列)
elem_c=4 (= pk 数)
```

---

## 5. 任务清单(见 TaskList,若丢失按此重建)

- [x] #1 手写 C-store layout `make_layout_gc`(C-layout 0)— 完成
- [x] #3 更新 probe_c_store(全 8 wave + 实测 ground truth)— 完成
- [ ] #5 钉死 step_k 语义(最小探针 Σ step_k==mma)— **先做**
- [ ] #2 方案A+step_k 主流水 + scale 乘法重构
- [ ] #6 重写 make_layout_sfb 对齐 C-layout 0(sfa 不动)
- [ ] #4 build + unit-scale 兜底 + random-scale 全绿

---

## 6. 历史遗留(供追溯,对本 kernel 无影响)

- 本 kernel 用普通 16x16x32 + 软件 scale,不走硬件 scaled-MFMA(方案见
  `SCALED_MMA_TODO.md`)。
- `PROGRESS.md` 里"一条 scaled 指令的一个 scale 覆盖整条 128K"的旧结论已被推翻,
  与本 kernel 无关。
- `probe_16x16x32.cu`(raw 指令 probe)给出的 C-layout 0 与 `probe_c_mma.cu`
  (经 opus adaptor)一致——即 opus swap_ab 的 C 输出方向就是 raw 指令方向。
  早先误以为 opus 会再转置一次,是错的。
