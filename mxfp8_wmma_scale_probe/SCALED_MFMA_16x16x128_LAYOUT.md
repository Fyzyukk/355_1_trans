# V_MFMA_SCALE_F32_16X16X128_F8F6F4 布局分析与验证方法

> 目标:钉死 gfx950 (CDNA4, wave64) 单条 scaled MFMA 指令
> `__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4` 的
> **数据布局**、**scale 布局**、以及**如何调用**。
> 所有结论均由本目录下的 probe 在真机 (MI355X, gfx950) 实测得出,非推测。

---

## 0. 名字澄清

- 用户/同事口中的 `V_WMMA_SCALE_F32_16X16X128_F8F6F4` 是 **RDNA (gfx1250, wave32)** 的 WMMA。
- 本机 gfx950 是 **CDNA4, wave64**,对应实存指令是 **MFMA**:
  `__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4`(opus 封装:`mfma_f32_16x16x128_fp8_fp8`)。
- wave64 下 `16*128/64 = 32`,一个 lane 持有 32 个 fp8 = `i32x8` = 256 bit VGPR。

---

## 1. 最终结论(TL;DR)

**数据布局 (A `[M=16,K=128]`,BLOCK 模型):**
```
lane L:  M-row  = L % 16
         K-block= L / 16      (每 block = 连续 32 个 K)
         持有 = A[row][ blk*32 : blk*32+32 ]   连续 32 个 fp8
```
| lane | row | K 范围 |
|---|---|---|
| 0  | 0 | [0,32)  |
| 16 | 0 | [32,64) |
| 32 | 0 | [64,96) |
| 48 | 0 | [96,128)|
| 1  | 1 | [0,32)  |

B `[K=128,N=16]` 对称:`lane L -> col=L%16, K-block=L/16`。

**scale 布局:** 每 lane 持 1 个 E8M0,作用于该 lane 的 (row/col, 32-K block)。
一条指令共吃 **A_scale[16 行][4 block] = 64 个** + **B_scale[16 列][4 block] = 64 个**。
scale 通过 **per-lane VGPR** 供给,不是一个标量广播到全 128K。

**累加器 C 布局:** `lane L, reg i (0..3)` -> `C[m = L%16][n = (L/16)*4 + i]`。

**E8M0 编码:** `actual_scale = 2^(byte - 127)`。`127 = ×1`(不缩放),`0 ≈ 2^-127 ≈ 0`(几乎乘没)。

---

## 2. 如何调用

### opus 封装(推荐)
```cpp
opus::mfma_f32_16x16x128_fp8_fp8 mma;
auto c = mma(v_a, v_b, v_c, scale_a, scale_b);                 // op_sel 默认 0
// 显式选字节:
auto c = mma(v_a, v_b, v_c, scale_a, scale_b,
             opus::number<osa>{}, opus::number<osb>{});
```
| 位 | 参数 | 类型 | 大小 | 含义 |
|---|---|---|---|---|
| 1 | v_a | `fp8x32_t`(=`i32x8`) | 256 bit | A,32 fp8/lane |
| 2 | v_b | `fp8x32_t` | 256 bit | B,32 fp8/lane |
| 3 | v_c | `fp32x4_t` | 128 bit | 累加器输入,4 float/lane |
| 4 | scale_a | `int` | 32 bit | 打包 4 个 E8M0 字节 |
| 5 | scale_b | `int` | 32 bit | 打包 4 个 E8M0 字节 |
| 6 | op_sel_a | `number<0..3>` 编译期 | — | 选 scale_a 的第几字节 |
| 7 | op_sel_b | `number<0..3>` 编译期 | — | 选 scale_b 的第几字节 |

返回 `fp32x4_t`。

### 底层 builtin(9 参)
```cpp
c = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
      i32x8 a, i32x8 b, fp32x4 c,     // 1,2,3
      int fmt_a, int fmt_b,            // 4,5 格式码 fp8=0 bf8=1 fp4=4 (编译期立即数)
      int op_sel_a, int scale_a,       // 6,7
      int op_sel_b, int scale_b);      // 8,9
```

### 不缩放 / 带 scale
```cpp
// 不缩放:必须 127,不能 0!
mma(v_a, v_b, v_c, 0x7F7F7F7F, 0x7F7F7F7F);   // 4 字节全 127
mma(v_a, v_b, v_c, 127, 127);                 // op_sel=0 只读 byte0=127 亦可

// 带 scale:每 lane 传自己 block 的 E8M0 放 byte0,op_sel=0
int scale_a = (int)sfa[lane%16][lane/16];     // (row, blk) 的 E8M0
mma(v_a, v_b, v_c, scale_a, scale_b);         // op_sel_a=op_sel_b=0
```

### ⚠️ 两个致命坑
1. `mma(v_a,v_b,v_c, 0, 0)` **错**:两个 0 落到 scale=0,`2^-127≈0`,结果被乘没。不缩放要传 127。
2. scale 放 byte0 就必须 **op_sel=0**。若 op_sel=1 硬件去读 byte1(=0)→ 结果乘没,即"错位"。
   op_sel 非 0 只在"一个 int 塞了多个 block 的 scale、想按字节选"时才用;
   当前 per-lane 布局每 lane 只需 1 个 scale,byte0 + op_sel=0 是标准。

### ds_read 次数
每 lane 32 fp8 = 256 bit,`ds_read_b128` 一次 16 fp8 → **每 lane 2 次 b128**
(K[0,16) + K[16,32) 连续两段),拼成一个 `fp8x32_t`。

---

## 3. 为什么 scale 是 [16][4] 而不是"每 lane 4 个"

这是最容易混的点。区别在指令的 K 宽度:

- **16x16x32**(旧):K 只有 32,一个 lane 8 个数据,要沿 K **迭代/repeat 4 次**才吃完 128,
  每次配 1 个 scale → 直觉"每 lane 4 个 scale"来源于此。
- **16x16x128**(本指令):**一条吃满 K=128,不 repeat**。
  一行的 128 个 K 被切给 **4 个不同 lane**(`L/16=0/1/2/3`),每 lane 连续 32 = 一个 microscale block,
  只需它那 **1 个** block 的 scale。所以"第一行的 4 个 K-block"不在同 4 个 lane,
  而在 lane 0/16/32/48,各持 `scale_a[row0, blk0/1/2/3]` 四个**不同** scale。

一条指令 = 16 行 × 4 block = 64 个 scale,一次完成 4 个 kg 的缩放 = "一条指令按 kg 做 scale"。

---

## 4. 验证方法论(如何钉死一条 MMA 的 layout)

核心难点与对策:

### 难点:内积对置换不变
MMA 沿 K 规约。只要 A、B 用**同一套** slot↔K 置换,内积不变 →
**光靠"数值对得上"无法唯一确定 slot↔K 的绝对映射**(见 `probe_klayout.cu`:3 种候选全 match,无效)。

### 对策 1:用 scale 做"探针",而非用数据位置
scale 是 per-lane 的、且线性作用于输出 → 可当作**可寻址的扰动源**:
boost 某一个 lane / 某一字节,看**哪个输出变、变多少**,反推该 lane 覆盖的 (row, block, 元素数)。

### 对策 2:构造"布局无关"的输入
让输入值只依赖逻辑坐标、且每个 slot 都相同(如 A=B=**全 1.0**),
这样"怎么把数据摆进寄存器"不影响数值,**排除布局假设带来的污染**。
(反例:`probe_data_layout.cu`/`probe_kset.cu` 在 kernel 里按 BLOCK 假设填数据,
若真实布局不同就错位,产出乱象 delta=256 —— 带假设的 probe 不可信,已作废。)

### 对策 3:单点扰动 + 增量幅度判据
"一个 lane 贡献几个元素" = boost 该 lane scale ×2 后**唯一变化输出线的增量**:
- delta = 32 → 该 lane = 一个完整 32-K block(**BLOCK 模型**)
- delta = 8  → 该 lane 每 block 8 个、跨 4 block(**REP4 模型**)

这是**布局无关**的判据(A=B 全 1,每 slot 都是 1),最干净。见 `probe_count.cu`。

### 对策 4:防常量折叠
scale 若为编译期常量,编译器可能把它当 F32 常量重解释,改变编码。
用 `volatile int` 拷一手强制走运行时路径(所有 probe 都这么做)。

---

## 5. Probe 清单与实测结果

| 文件 | 作用 | 结论 |
|---|---|---|
| `probe_count.cu` | **布局无关**测每 lane 元素数(决定性) | **delta=32 → BLOCK 模型** |
| `probe_scale_map.cu` | 单 lane 打热,测 lane→(row/col,block) | lane L → (L%16, L/16),4 block 各独立 scale |
| `probe_routing.cu` | E8M0 编码 + op_sel 选字节 | `2^(b-127)`;op_sel 选 byte(**单标量误导版**,见下) |
| `probe_klayout.cu` | 数据 K 排布候选 | 全 match,证明"内积无法定位"→ 需 scale 探针 |

> 已删除的反面教材(供记录):`probe_data_layout.cu` / `probe_kset.cu` / `probe_scale_kg.cu`
> —— 在 kernel 里按 BLOCK 假设填数据,若真实布局不同即错位,产出乱象(delta=256)。
> 教训见 §4 对策 2:probe 不能带布局假设。`probe_scale_kg.cu` 的功能已被 `probe_scale_map.cu` 覆盖。

### 关键实测片段
```
# probe_count.cu —— boost 一个 lane scale x2,delta=32:
  L= 0 (row0,blk0): delta=32 on 16 outputs, first at (m=0,n=0)
  L=16 (row0,blk1): delta=32 on 16 outputs, first at (m=0,n=0)
  L=48 (row0,blk3): delta=32 on 16 outputs, first at (m=0,n=15 for L=63)
  => delta=32 => BLOCK 模型确认

# probe_scale_map.cu —— per-block byte0=127+g,op_sel0 vs op_sel1:
  op_sel0 C[0][0]=480 (=32*(1+2+4+8))   op_sel1 C[0][0]=128 (读 byte1=127)
  => 4 个 block 独立 scale;op_sel 正交选字节
```

### ⚠️ 关于 `probe_routing.cu` 的更正
该 probe 给**所有 lane 传同一个标量** scale,因此只能看到 op_sel 选字节的效果,
误得"一条指令只有一个 scale 覆盖全 128K"。**真相**是 scale per-lane,
不同 K-block 从不同 lane(L/16)取 scale —— 由 `probe_scale_map` / `probe_count` 修正。
教训:测 per-lane 资源必须**逐 lane 给不同值**,否则看不到 lane 维度。

---

## 6. 编译运行
```sh
hipcc probe_count.cu -I. -I/root/workspace/aiter/csrc/include \
      -std=c++17 -O3 --offload-arch=gfx950 -o probe_count.exe
./probe_count.exe
```
