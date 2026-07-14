# 单-wave→多-wave layout 推广 & 逐层验证方法论(通用可复用)

> 目标:已知一条 MMA 指令的单-wave layout(A/B/C 的 `(lane,slot)->(m/n,k)`,
> 通常由 `mxfp8_wmma_scale_probe/` 的真机 probe 钉死),要把它推广成
> 一个多-wave 的 block-tile GEMM(例:128×128×128,8 wave = M4×N2)。
> 推广是**机械的**,但极易在"LDS swizzle / fragment 转置方向"上错位。
> 本方法论给出**不靠肉眼、不碰 MMA 数值**就能逐层钉死对错的流程。
>
> 姊妹文档:`mxfp8_wmma_scale_probe/MMA_LAYOUT_PROBING_METHODOLOGY.md`
> (怎么用真机 probe 钉死**单条指令**的 layout)。本文接力:钉死之后,
> 怎么验证**手写的多-wave 搬运 layout** 与硬件一致。

---

## 0. 适用前提

- 单-wave 指令 layout 已由真机 probe 钉死(见 probe 方法论 §5.2/§5.3)。
- kernel 用 opus 的 `make_layout_*` 手写了 6 个搬运 layout:
  `ga/sa/ra`(A: global→LDS→reg)、`gb/sb/rb`(B),外加 C-store(`partition_layout_c`)。
- 要回答的问题:**这些手写 layout 与硬件真实 layout 是否逐元素一致?**

---

## 1. 推广的几何(先对数字,再验布局)

从单-wave `W_M×W_N×W_K` 推到 block-tile `B_M×B_N×B_K`、`T_M×T_N` 个 wave:
```
E_M = (B_M or HALF_B_M) / (W_M * T_M)   # M 方向每 wave repeat 次数
E_N = (B_N or HALF_B_N) / (W_N * T_N)   # N 方向 repeat
E_K = B_K / (W_K * T_K)                 # K 方向 repeat(把 W_K 拼满 B_K)
```
`make_tiled_mma(seq<E_M,E_N,E_K>, seq<T_M,T_N,T_K>, seq<W_M,W_N,W_K>, adaptor)`
一次 `mma(v_a,v_b)` 调用内部自动循环 `E_K*E_M*E_N` 次子-MMA 累加。
**先确认 `v_a/v_b/v_c` 尺寸 == `tile_a_len/tile_b_len/tile_c_len`**(编译期 static_assert
会挡住尺寸错;能编过 ≠ 布局对,但尺寸对是第一道关)。

---

## 2. 核心洞见:数值抹平位置 → 必须验"地址",而非"数值"

内积对 K 置换不变(见 probe 方法论 §1),所以**跑 GEMM 看数值对错永远无法定位**
是哪一层搬运错了。而且 LDS 是 swizzle 的,直接比 offset 数值也没意义(坐标系不同)。

**正确的判据是"往返闭合 + 对齐硬件":**
1. **round-trip(往返闭合)**:A 走 `ga→sa`(写 LDS)再 `ra`(读 LDS),
   若同一个逻辑元素 `(m,k)` 写进去又读出来能对上,则 sa/ra **互相自洽**。
2. **oracle(对齐硬件)**:opus 的 canonical `mma.layout_a/b/c(stride, p_coord)`
   是"硬件要求的正确布局"。手写 layout 的往返结果必须逐元素等于它。
3. **oracle 自检**:canonical 本身也可能对某条指令是错的(实例见本目录 DEBUG 文档的
   C-store!),所以**先拿 canonical 和真机 probe 表对一遍**,确认 oracle 可信,再拿它当标尺。

三者缺一不可。只有 round-trip 会漏"整体系统性偏移";只有 oracle 会漏"opus 对该指令的 bug"。

---

## 3. 探针写法(纯 address 记账,全 host 可读)

数据用 fp8 存不下指纹,所以**不搬数据、只搬地址**。三个独立小 kernel:

### 3.1 A/B round-trip 探针(验 ga/sa/ra、gb/sb/rb)

```
WRITE 侧(复现 async_load 的搬运):
  ga_off = layout_to_offsets<VEC_A_GLOBAL>(u_ga)   # 源:A tile 内元素偏移 = m*B_K + k
  sa_off = layout_to_offsets<VEC_A_GLOBAL>(u_sa)   # 目的:LDS 偏移
  for each issue i, each j in [0,VEC_A_GLOBAL):
    lds_to_mk[ sa_off[i] + lane_id*VEC_A_GLOBAL + j ] = ga_off[i] + j
    #                       ^^^^^^^^^^^^^^^^^^^^^^^^ 关键!见 §4 坑1
READ 侧(wave0,复现 ra 读 LDS):
  ra_off = layout_to_offsets<VEC_A>(u_ra)
  for each issue i, each j in [0,VEC_A): slot=i*VEC_A+j
    ra_mk[lane][slot] = lds_to_mk[ ra_off[i] + j ]      # 解码回 (m,k)
ORACLE:
  mma = make_tiled_mma<...>(...同 kernel...)
  p_coord_a = (wave_id_m, lane%W_M, 0, lane/W_M)        # 见 §4 坑2
  ca_off = layout_to_offsets<VEC_A>( partition_layout_a<VEC_A>(mma, {B_K,1}, p_coord_a) )
  canon_mk[lane][slot] = ca_off ...                     # 直接 = m*B_K + k
比对:ra_mk vs canon_mk 逐元素;并统计 LDS 覆盖数(应 == tile 元素数,验双射)。
```
B 路径完全对称(`layout_b`, p_coord_b = `(wave_id_n, lane%W_N, 0, lane/W_N)`)。

### 3.2 C-store 探针(验 partition_layout_c + fragment 序)

C 不经 LDS(寄存器→global),只需:
```
u_gc = partition_layout_c<VEC_C>(mma, {STRIDE_C,1}, p_coord_c)   # 同 kernel store 时
gc_off = layout_to_offsets<VEC_C>(u_gc)
c_mn[lane][slot] = gc_off[i]+j   # = m*STRIDE_C + n
判据1(双射):所有 (lane,frag) 覆盖 128×128 一次且仅一次(missed=dup=oob=0)。
判据2(对齐 probe):fragment i 的 (m,n) == probe 表预测(见 §4 坑3 的转置方向!)。
```

---

## 4. 三个致命坑(实战全踩过,记下来免得再踩)

**坑1 — buffer_load_lds 的 lane 散布**:CDNA 上 `async_load` 到 LDS 时,
`u_sa` 不含 `lane_id`(mxfp8/block_scale 风格),硬件按 lane 自动把每 lane 的
`VEC` 个元素散到 `dst_base + lane*VEC`。探针 WRITE 侧**必须手动加 `lane_id*VEC`**,
否则所有 lane 写同一处、LDS 覆盖只有 1/64(实战第一版就是 256 而非 16384)。
(注:fp8_gemm 风格把 lane_id 显式放进 sa 的 p_coord,则不需要额外加。看 kernel 用哪种。)

**坑2 — canonical p_coord 的顺序**:`layout_a` 的 p_coord 顺序 =
`(tile_m, grpm_a, tile_k, grpk_a)` = `(wave_id_m, lane%W_M, 0, lane/W_M)`。
顺序错会生成一个"看起来合法但错位"的 oracle,导致假阴性。以 `dim_a()` 里
`p_dim` 出现的顺序为准(见 opus.hpp `mfma_adaptor::dim_a/b/c`)。

**坑3 — C fragment 的转置方向(决定性易错点)**:同样是"每 lane 4 个 C 寄存器",
16x16x32 与 16x16x128 的 `(gk,i)→(m,n)` 绑定方向**相反**(probe 方法论 §5.3):
| 指令 | C 映射 | 每 lane 4 个 C |
|---|---|---|
| 16x16x32  | `C[gk*4+i][row]` | 同 1 **列**、连续 4 **行** |
| 16x16x128 | `C[row][gk*4+i]` | 同 1 **行**、连续 4 **列** |
opus 的 `mfma_adaptor_swap_ab::shape_c/dim_c` 是**通用**式,对 16x16x32 会排成
"连续 4 列"——**方向错**。这类 opus-对该指令-有 bug 的情况,正是 §2 判据3
(oracle 自检)存在的理由。

---

## 5. Checklist(照做)

```
[ ] 算 E_M/E_N/E_K,确认 v_a/v_b/v_c 尺寸 == tile_*_len(编译期)
[ ] oracle 自检:canonical layout_a/b/c vs 真机 probe 表 → 0 mismatch 才可信
[ ] A round-trip:ra vs canonical layout_a → 0 mismatch + LDS 全覆盖
[ ] B round-trip:rb vs canonical layout_b → 0 mismatch + LDS 全覆盖
[ ] C-store:双射(128×128 覆盖一次) + fragment 序对齐 probe(注意转置方向!)
[ ] WRITE 侧记得 lane*VEC 散布(坑1);p_coord 顺序对(坑2);C 方向对(坑3)
```

配套探针源码(本目录,可直接复用改尺寸):
`probe_ra_roundtrip.cu`、`probe_rb_roundtrip.cu`、`probe_c_store.cu`。

---

## 6. 一句话总结

> **单-wave layout 推广到多-wave 是机械的,但对错要靠"地址往返闭合 + 对齐 canonical oracle"
> 三重判据来验,而不是跑 GEMM 看数值;canonical 本身也可能对某条指令错,所以先拿它和真机
> probe 表自检。三个坑必踩:lane*VEC 散布、p_coord 顺序、C fragment 转置方向。**
