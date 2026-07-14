# Matrix 指令 (MFMA/WMMA) Layout 方法论 + 实测仓库

> 适用:任意 gfx MFMA/WMMA 指令(带或不带 scale)。
> 目标:不靠文档/推测,用真机实测钉死一条指令的
> **A/B 数据布局**、**C 累加器布局**、**scale 布局**、**编码语义**。
> 本文是从 `V_MFMA_SCALE_F32_16X16X128_F8F6F4` 验证过程中抽出的通用流程,
> 并随后扩到 16x16x32 等指令,汇成一个可持续积累的 layout 仓库。
>
> **两部分**:
> 1. **方法论**(§0~§5.1、§6~§7):怎么设计 probe、避坑、钉死一条指令。
> 2. **实测仓库**(§5.2 实例 + §5.3 速查表):已钉死指令的 layout,按指令查即用。
>    新钉死一条就往 §5.3 加一行。

---

## 0. 先建立坐标系(动手前必做)

一条指令由 6 个数字定义:`M, N, K, warp_size, elem/lane, dtype`。先算清楚:

```
elem_a = M*K / warp_size      # 每 lane 持有多少 A 元素
elem_b = N*K / warp_size      # 每 lane 持有多少 B 元素
elem_c = M*N / warp_size      # 每 lane 持有多少 C 元素
reg_bits = elem * sizeof(dtype)*8   # 每 lane 寄存器位宽
```
例:16x16x128 fp8, wave64 → elem_a=32(256bit=i32x8), elem_c=4(fp32x4)。

**要解的未知量**(每条指令都是这三张映射表):
1. `A: (lane, slot_a) -> (m, k)`
2. `B: (lane, slot_b) -> (k, n)`
3. `C: (lane, slot_c) -> (m, n)`
4. (若带 scale)`scaleA: (lane) -> (m, k-block)`,`scaleB: (lane) -> (n, k-block)`

---

## 1. 核心难点:内积对置换不变

MMA 沿 K 规约:`C[m][n] = Σ_k A[m][k]*B[k][n]`。
**只要 A 和 B 用同一套 slot↔k 置换,C 不变。**
→ 光看"数值对不对"**永远无法唯一确定** slot↔k 的绝对位置。

**推论:必须引入一个"可寻址的、不被规约抹平"的探针。**
- 有 scale 的指令:**scale 就是最好的探针**(per-lane 可寻址,线性作用于输出)。
- 没有 scale 的指令:用 **单点非零 + 输出位置** 反推(见 §5),
  或更强的 **"唯一指纹"构造法**(见 §5.1,推荐)。

### ⚠️ 假阳性陷阱(本方法最容易翻车的地方)

**只要候选 C-layout 是 (lane,slot)→(m,n) 的一个置换,而参考矩阵里有重复值,
它就可能"碰巧全部对上",给出假 MATCH。** 实测教训:用
`Aval/Bval ∈ {0,1}` 生成的 `C` 只有少数几种值(如 0/10/11),结果
**6 个 C-layout 候选 × 2 个 A/B-layout 候选全部 MATCH** —— 完全无法区分。

两层原因叠加:
1. **A/B 维**:内积对 slot↔k 置换不变(§1),A/B-layout 本就无法靠 C 数值区分。
2. **C 维**:参考值域太小 → 不同 (m,n) 数值碰撞 → 错误的 C-layout 也蒙对。

**破解:让 `C[m][n]` 每个 cell 取唯一值(见 §5.1)。** 一旦 C 是 0..M·N−1 的双射,
错误 layout 必然在某个 cell 露馅,唯一正确的 layout 才会 MATCH。

---

## 2. 四条铁律(probe 设计原则)

| 铁律 | 原因 | 反例后果 |
|---|---|---|
| **A. 输入布局无关** | 让数值只依赖逻辑坐标,不依赖"怎么摆进寄存器" | kernel 里按假设填数据 → 布局若不同就错位,产出乱象(delta=256) |
| **B. per-lane 逐个给不同值** | 要测 lane 维资源,必须让每 lane 可区分 | 所有 lane 给同一标量 → 看不到 lane 维,误判"全局单值" |
| **C. 单点扰动 + 看增量** | 隔离单个自由度,增量幅度/位置直接给出答案 | 多点同时动 → 无法归因 |
| **D. 防常量折叠** | 编译期常量 scale 会被当 F32 重解释,改变编码 | 用 `volatile` 拷一手强制运行时 |

铁律 A 的黄金输入:**A=B=全 1.0**。每个 slot 都是 1,无论怎么排,数值都一样,
布局假设被彻底排除。扰动全部来自 scale/单点,干净可归因。

---

## 3. 标准流程(5 步)

### Step 1 — 确认指令存在 + 基线值
A=B=全 1.0,scale=unit(E8M0=127)。跑一条,读全部 C。
**基线 = K**(K 个 1 相乘累加)。对不上说明 dtype 编码/调用姿势错,先修这里。

### Step 2 — 解 C 布局:`(lane, slot_c) -> (m, n)`
带 scale:boost **单个 lane** 的 scale ×2,看**哪个输出线**变 → 该 lane 对应的 (m 或 n)。
扫所有 lane,拼出 C 映射。
不带 scale:见 §5(单点数据法)。

### Step 3 — 解每 lane 元素数 + 数据 block 归属(BLOCK vs REP4)
A=B=全 1,boost 单 lane scale ×2,量**唯一变化输出线的增量 delta**:
```
delta == elem_a         → 该 lane 的全部元素在同一连续 K 段(BLOCK 模型)
delta == elem_a / n_blk → 该 lane 每 block 只有一部分,跨多 block(REP/interleave 模型)
```
delta 幅度是**布局无关判据**,最可靠。这一步同时定死 `lane -> k-block`。

### Step 4 — 解 scale 布局 + 编码
- **编码**:packed int 四字节全填 b,扫 b∈[125,129],验 `C = K * 2^(2*(b-127))` → 确认 `2^(b-127)`。
- **粒度/路由**:给不同 block 不同 scale(per-lane byte0 = 127+blk),
  若 `C = Σ_blk 2^blk * (blk_size)` → 每 block 独立 scale(per-kg)。
- **op_sel 语义**:同一份数据,op_sel=0 vs 1 → 证明它是"选 packed int 的第几字节",与 block 路由正交。

### Step 5 — 解 A/B 绝对位置(可选,需要时)
用 §5 单点法:A 只在单个 (m0,k0)=1,其余 0,B=全 1 →
`C[m0][n] = 1` 且唯一;读出它落在哪个 (lane,slot) → 得 A 的绝对映射。
(注意:此步 kernel 装数据也不能带假设,靠"全数组可寻址 + 逻辑坐标"喂。)

---

## 4. 判据速查表

| 观测 | 结论 |
|---|---|
| 基线 C == K | 调用/编码正确 |
| boost lane L → 只有列 n* 变 (A=B=1) | scaleA lane L 管 row/col = 从 n* 反推 |
| delta == elem/lane | BLOCK 模型(每 lane 一段连续 K) |
| delta == elem/(lane·nblk) | 分散模型(REP/interleave) |
| 不同 block 不同 scale → C 变化 | scale 按 k-block 粒度(per-kg) |
| op_sel 0/1 结果不同、与数据正交 | op_sel = packed int 字节选择器 |
| 单标量 scale 改 op_sel → C 分档变 | (误导!)说明你没逐 lane 给值,回 Step 见铁律 B |

---

## 5. 无 scale 指令怎么办(单点数据法)

没有 scale 探针时,用**数据的单点 + 输出位置**代替:
- **解 C**:A[m0][*]=1(单行),B=全 1 → 只有 row m0 的输出非零 → 定位 m。
  再 A=全 1,B[*][n0]=1 → 定位 n。
- **解 A 的 (lane,slot)->(m,k)**:A 只在 (m0,k0)=1 → `C[m0][n]=1`;
  但要知道 k0 落在哪个 lane/slot,需配合"逐 slot 置 1 扫描 + 观察哪次生效",
  或借助 `ds_read` 前的寄存器 dump(把 a_reg 直接写回 global 检查)。
- **最省事**:直接 dump 寄存器。kernel 里把 `a_reg[i]` 按已知全局地址写回,
  host 比对 → 直接读出 (lane,slot)->(m,k),不经过 MMA。

> 寄存器 dump 法对任何指令都通用,但要求你能正确 load(先有 load layout 假设);
> scale 探针法则不需要正确 load(A=B=1),更适合"从零"起步。

---

## 5.1 唯一指纹构造法(解 C-layout 的首选,无 scale 也适用)

**核心思想:构造 A、B 使得 `C[m][n] = m·N + n`(0..M·N−1 全唯一)。**
这样每个 C 元素就是它自己坐标的"指纹",读出 `raw[lane,slot]` 的值就直接反解出 `(m,n)`,
**一次 kernel 启动就拿到完整 C 映射表**,且天然免疫 §1 的假阳性陷阱。

### 怎么造出 `C[m][n]=m·N+n`

用两个 K 点即可(其余 K 填 0),让内积恰好拆成"高位 + 低位":
```
A[m][0] = m,   A[m][1] = 1,   其余 A[m][k]=0
B[0][n] = N,   B[1][n] = n,   其余 B[k][n]=0
C[m][n] = A[m][0]·B[0][n] + A[m][1]·B[1][n] = m·N + n     ✅ 唯一
```
- 只有 k=0,1 非零 → **A/B 的 slot↔k 置换与结果无关**(只要 A、B 用同一套映射),
  因此这一步**不需要**先知道 A/B 的 K-layout,专注解 C。
- 值域要 fp32/整数可精确表示。fp8(e4m3)做输入时,`m,n,N` 都要能被 e4m3 精确编码
  (小整数 0..16 没问题;更大范围改用能精确表示的量,或换 bf16/fp16 输入探)。

### 读法

`raw[lane*elem_c + i]` 的值 = `m·N + n` → `m = v/N, n = v%N`。
把 (lane, slot i) → (m,n) 全表打印,再和候选 layout 逐一比对,**只会有一个 MATCH**。

### 为什么比 §5 单点法强

§5 单点法一次只点亮一个 cell,要扫 M·N 次;唯一指纹法**一发入魂**,
且直接给出人类可读的映射表,肉眼即可看出规律(见下面 16x16x32 实例)。

---

## 5.2 完整实例:钉死 `V_MFMA_F32_16X16X32_FP8_FP8`(无 scale)

一步步走一遍,展示唯一指纹法怎么"一发入魂"。配套代码 `probe_16x16x32.cu`。

### Step 0 — 坐标系

M=N=16, K=32, wave64:
```
elem_a = 16*32/64 = 8      # 每 lane 8 个 A(fp8) = 64bit long
elem_b = 16*32/64 = 8      # 每 lane 8 个 B
elem_c = 16*16/64 = 4      # 每 lane 4 个 C(fp32)
```
调用: `__builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(long a, long b, fp32x4 c, cbsz,abid,blgp)`。

### Step 1 — 先踩坑(反面教材)

第一版用 `A,B ∈ {0,1}` 的哈希填充 → `C` 只有 {0,10,11} 三种值 →
**6×2 个候选 layout 全 MATCH**(假阳性)。原因见 §1 的假阳性陷阱。

### Step 2 — 唯一指纹(§5.1)

改成 `C[m][n]=m*16+n`(A[m][0]=m,A[m][1]=1; B[0][n]=16,B[1][n]=n),
读回 `raw[lane*4+i]`,对 6 个候选 C-layout 比对 → **只有 C-layout 1 MATCH**。

### 实测结论(钉死的三张表)

```
lane L:  row = L%16 (0..15),  gk = L/16 (0..3)
         # row = lane 低 4 位 = 该 lane 的 M 行(A/C)或 N 列(B);
         # gk  = lane 高 2 位 = 该 lane 的分组号。在 A/B 里表示"负责哪段 K",
         #       在 C 里表示"产出哪个 4 行块"。同一个 L/16,两处物理含义不同(见 §5.3 约定)。

C: (lane L, slot i∈0..3) -> (m, n):
      n = row = L % 16                 ← 列 = lane 低 4 位
      m = gk*4 + i = (L/16)*4 + i      ← 行 = 分组号 ×4 + 寄存器槽
   即每个 lane 的 4 个 C 寄存器 = 同一列 n、连续 4 行 m。(swap_ab 转置 C 布局)

A: (lane L, slot i∈0..7) -> A[row][gk*8 + i]    # gk 段 = K 的第 gk 个 8-K 段,每 lane 连续 8-K
B: (lane L, slot i∈0..7) -> B[gk*8 + i][row]    # 4 个 gk 段拼满 K=32
```

### 举例:单 wave 里 **lane 0** 到底负责哪些元素

`L=0` → `row=0, gk=0`(gk 在 A/B 里是"第 0 个 K 段",在 C 里是"第 0 个行块"):

- **A 寄存器(8 个 fp8)**:`a_reg[i] = A[0][0*8+i] = A[0][i]`,i=0..7
  → lane0 拿 **A 矩阵第 0 行的 K=0..7** 这 8 个元素。
- **B 寄存器(8 个 fp8)**:`b_reg[i] = B[0*8+i][0] = B[i][0]`,i=0..7
  → lane0 拿 **B 矩阵第 0 列的 K=0..7** 这 8 个元素。
- **C 寄存器(4 个 fp32)**:`n = row = 0`,`m = gk*4+i = i`
  → lane0 输出 **C[0][0], C[1][0], C[2][0], C[3][0]**
  (第 0 列、第 0~3 行这 4 个 C 元素)。

对照几个别的 lane 帮助建立直觉:

| lane | row | gk | kk | A 元素 | B 元素 | C 输出(4 个) |
|---|---|---|---|---|---|---|
| 0  | 0  | 0 | 0 | A[0][0..7]   | B[0..7][0]   | C[0..3][0] |
| 1  | 1  | 0 | 0 | A[1][0..7]   | B[0..7][1]   | C[0..3][1] |
| 15 | 15 | 0 | 0 | A[15][0..7]  | B[0..7][15]  | C[0..3][15] |
| 16 | 0  | 1 | 1 | A[0][8..15]  | B[8..15][0]  | C[4..7][0] |
| 48 | 0  | 3 | 3 | A[0][24..31] | B[24..31][0] | C[12..15][0] |

**读法**:同一 `row`(lane%16)的 4 个 lane(L, L+16, L+32, L+48)沿 K 分工
(各拿连续 8-K 的一段,拼满 32),它们的 C 输出覆盖同一列 n=row 的全部 16 行。

### 对写 kernel 的直接含义(为什么这一步值)

一个 lane 的 4 个 C 元素是 **4 个不同 M 行、同 1 个 N 列**。因此做 per-row/per-group
后缩放时:
- **A-scale 随 slot i 变**(4 个不同 m 行) → 必须逐 slot 取 `scale_a[gk*4+i]`;
- **B-scale 对 4 个 slot 相同**(同 n 列) → 只取一个 `scale_b[row]`。

用同一个索引同时查 A、B scale 是错的 —— 这正是靠肉眼看这张表才能发现的。

---

## 5.3 Layout 仓库(已实测钉死的指令速查)

> 本节把本目录用上述方法钉死的指令 layout 汇成一张速查表,作为可持续积累的"仓库"。
> 新钉死一条指令就往这里加一行 + 一小节。所有结论均真机(MI355X, gfx950/CDNA4, wave64)实测。
>
> **符号约定**(先看懂这个,下面公式才不含糊):
> - `lane L`:wave 内的 lane 号,0..63。
> - `row = L % 16`:lane 的**低 4 位**。它同时是这条 lane 负责的 A 的 M 行 / B 的 N 列
>   (16 行/列正好用满低 4 位)。
> - `gk = L / 16`:lane 的**高 2 位**,取值 0..3(因为 64 lane / 16 = 4)。
>   **`gk` = "这条 lane 在被 16 整除后落在第几组"**,是硬件把 64 条 lane 切成 4 组的组号。
>   它的**物理含义随矩阵不同**:
>     - 在 **A/B(输入)** 里,`gk` 是 **K 方向的分段号**:4 组 lane 各吃 K 的一段,拼满整个 K。
>       (16x16x32: 每段 8 个 K;16x16x128: 每段 32 个 K)
>     - 在 **C(累加器)** 里,`gk` 是 **输出的行块号(或列块号)**:4 组 lane 各产出
>       M(或 N)的一个连续 4 元素块。
>   → 同一个 `gk=L/16` 在输入端表示"我负责哪段 K",在输出端表示"我产出哪块行/列";
>     这是因为硬件用同一批高位 lane 复用了输入和输出的分组,不是两个不同的量。
> - `slot i`:该 lane 在对应矩阵寄存器数组里的下标(A/B 是 0..elem-1,C 是 0..3)。

### 速查总表

| 指令 | M×N×K | dtype | elem_a/b/c per lane | scale | 详见 |
|---|---|---|---|---|---|
| `mfma_f32_16x16x32_fp8_fp8` | 16×16×32 | fp8→fp32 | 8 / 8 / 4 | 无 | §5.2 / `probe_16x16x32.cu` |
| `mfma_scale_f32_16x16x128_f8f6f4` | 16×16×128 | fp8→fp32 | 32 / 32 / 4 | 有(E8M0) | 下方 / `SCALED_MFMA_16x16x128_LAYOUT.md` |
| `mfma_f32_16x16x128_fp8_fp8`(不带 scale 用法) | 16×16×128 | fp8→fp32 | 32 / 32 / 4 | 无(传 127) | 下方(同布局) |

### A) `V_MFMA_F32_16X16X32_FP8_FP8`(无 scale)

见 §5.2 完整推导。一句话:
```
A: (L, i∈0..7) -> A[row][ gk*8 + i ]        # 每 lane 连续 8-K
B: (L, i∈0..7) -> B[ gk*8 + i ][row]
C: (L, i∈0..3) -> C[ gk*4 + i ][row]        # 每 lane 4 个 = 同列 row、连续 4 行
```
lane0 → A[0][0..7], B[0..7][0], 输出 C[0..3][0]。4 个 lane(L,L+16,L+32,L+48)沿 K 分工拼满 32。

### B) `V_MFMA_SCALE_F32_16X16X128_F8F6F4`(带 E8M0 scale)

wave64 下 `16*128/64=32`,一 lane 持 32 fp8 = `i32x8` = 256bit。**BLOCK 模型**(每 lane 一段连续 32-K):
```
A: (L) -> A[row][ gk*32 : gk*32+32 ]        # 每 lane 连续 32-K = 一个 microscale block
B: (L) -> B[ gk*32 : gk*32+32 ][row]
C: (L, i∈0..3) -> C[row][ gk*4 + i ]        # 注意: 与 16x16x32 转置方向不同(见下"⚠️对比")
scale: 每 lane 1 个 E8M0,作用于 (row/col, 该 gk block);
       一条指令共吃 A_scale[16行][4block]=64 + B_scale[16列][4block]=64。
```
lane0 → A[0][0:32], B[0:32][0], scale=(row0,blk0), 输出 C[0][0..3]。
同一行的 4 个 K-block 分给 lane 0/16/32/48,各持**不同** scale → 一条指令按 kg 缩放 4 个 block。

**E8M0 编码**:`actual = 2^(byte-127)`。不缩放传 **127**(不是 0!传 0 会 `2^-127≈0` 把结果乘没)。
调用/op_sel/ds_read 细节见 `SCALED_MFMA_16x16x128_LAYOUT.md` §2。

### C) `V_MFMA_F32_16X16X128_FP8_FP8`(同尺寸不带 scale)

数据/累加器布局与 B) **完全相同**(scale 是 MMA 之外的旁路,不改变 A/B/C 排布)。
差别仅在调用:用普通 dispatch(无 scale 参数),或走 scaled builtin 但 4 字节全填 127。

### ⚠️ 16x16x32 与 16x16x128 的 C 布局对比(易错点)

两条指令的 C 都是"每 lane 4 个寄存器",但 **m/n 与 (gk,i) 的绑定方向相反**:

| 指令 | C 映射 | 每 lane 4 个 C 覆盖 |
|---|---|---|
| 16x16x32   | `C[gk*4+i][row]` | **同 1 列 n=row、连续 4 行 m** |
| 16x16x128  | `C[row][gk*4+i]` | **同 1 行 m=row、连续 4 列 n** |

> 迁移 kernel 时若照抄 store/scale 索引会错位。写后缩放尤其要注意:
> 16x16x32 里"4 个 C = 4 个不同行" → A-scale 逐 slot 变、B-scale 4 slot 同;
> 16x16x128 里"4 个 C = 4 个不同列" → 反过来,B-scale 逐 slot 变、A-scale 同。
> (以各自实测表为准;这类方向差异正是必须真机 probe、不能照搬文档的原因。)

---

## 6. Checklist(照着做)

```
[ ] 算 elem_a/b/c、reg 位宽
[ ] Step1: A=B=1, scale=127 → 基线 == K ?
[ ] Step2: 单 lane boost → 拼 C 映射 (lane,slot)->(m,n)
[ ]   无 scale 时首选: 唯一指纹 C[m][n]=m*N+n (§5.1) → 一发拿到 C 全表
[ ]   ⚠️ 参考值必须唯一! 值域太小(如 {0,1})→ 假 MATCH 全中(§1 陷阱)
[ ] Step3: 单 lane boost 量 delta → BLOCK / 分散? 定 lane->k-block
[ ] Step4: 扫字节值验 2^(b-127); 分 block 给值验 per-kg; op_sel 0/1 验字节选择
[ ] Step5(按需): 单点数据 / 寄存器 dump → A/B 绝对 (m,k)/(k,n)
[ ] 全程: A=B=1 保布局无关; 逐 lane 不同值; 单点扰动; volatile 防折叠
```

---

## 7. 一句话总结

> **数据位置被内积抹平,所以要用"可寻址的探针"(scale、单点、或唯一指纹)去戳,
> 用"布局无关的输入"(全 1 / 只点亮 2 个 K)保证戳出来的信号干净,
> 用"值的唯一性"防假阳性,
> 用"增量的幅度和位置"反推每个 lane 到底管哪些 (m,n,k) 和哪个 scale。**

> 一句话选型:
> - **有 scale** → scale 探针(§2~§4)。
> - **无 scale、要 C 布局** → 唯一指纹 `C=m·N+n`(§5.1),最快最稳。
> - **无 scale、要 A/B 绝对 (m,k)** → 寄存器 dump 或单点(§5)。
> - **任何时候** → 参考值务必唯一,否则置换型错误 layout 会全部假 MATCH(§1)。

配套实证见同目录 `SCALED_MFMA_16x16x128_LAYOUT.md`、`probe_16x16x32.cu`(唯一指纹实例) 与各 `probe_*.cu`。
```
