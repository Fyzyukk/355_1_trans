# block_scale_gemm_bpreshuffle 交接文档

本文档记录 B-preshuffle 变体的**已完成实现**、**关键推导结论**、以及**后续激进版方案**,
供新会话接续。上一会话上下文将满,故落盘。

## 0. 环境 / 构建 / 运行(已核实可用）

- opus 头文件真实路径:`/root/workspace/aiter/csrc/include/opus/opus.hpp`
  编译前:`export OPUS_INCLUDE_DIR=/root/workspace/aiter/csrc/include`
- 硬件 gfx950 在机,`/opt/rocm/bin/hipcc`（HIP 7.0）可就地编译运行。
- 目录:`/root/workspace/amdgcn_fyz/gcnasm/opus_gemm/block_scale_gemm_bpreshuffle/`
- 构建 / 运行:
  ```
  cd opus_gemm/block_scale_gemm_bpreshuffle
  export OPUS_INCLUDE_DIR=/root/workspace/aiter/csrc/include
  make -j
  ./build/gemm_a8w8_blockscale_bpreshuffle.exe -b 1 -m 1024 -n 1024 -k 1024 -v 1
  ```
- 基线（对照）:`opus_gemm/block_scale_gemm/`（未改动的原 kernel）。

## 1. 当前状态:已完成,数值全部 VALID

这是 **B-preshuffle 的最小正确变体**（"连续读、仍过 LDS"）。相对基线 `block_scale_gemm`,
只做了两处改动,其余流水线（LDS staging / ds_read / MFMA schedule / scale / waitcnt 计数）
**完全未动**,因此原本精细排布的软件流水线保持有效。

### 文件（均由基线拷贝改名而来）
```
block_scale_gemm_bpreshuffle/
├── Makefile                                            # OUT/符号名改 bpreshuffle
├── rebuild.sh
├── gemm_a8w8_blockscale_bpreshuffle_common.h           # kargs + traits（与基线内容相同）
├── gemm_a8w8_blockscale_bpreshuffle_kernel_template.hpp # kernel body（仅 gb_offset 一行改动）
├── gemm_a8w8_blockscale_bpreshuffle_kernel.cc          # device-only TU + host stub
├── gemm_a8w8_blockscale_bpreshuffle_host.cc            # host + shuffle_b() + CPU ref
└── README.md
```
注意:kernel 符号已从 `gemm_a8w8_blockscale_kernel` 改名为
`gemm_a8w8_blockscale_bpreshuffle_kernel`（template.hpp / kernel.cc / host.cc 三处一致）。
traits 结构体名仍是 `gemm_a8w8_blockscale_traits<>`（未改,内容与基线相同）。

### B' 布局定义（核心）
host 把 `B[N,K]`（行主序）重排成 K-panel:
```
B'[batch][kt][n][ki] = B[batch][n][kt*BLOCK_K + ki]     # 形状 [batch, K/BLOCK_K, N, BLOCK_K]
per-batch 线性下标 = (kt*N + n)*BLOCK_K + ki
```
每个 `BLOCK_K` 宽的 K-panel 内,某个 n 行是连续的;panel 内相邻 n 行间距 = `BLOCK_K`。

### 两处改动
1. **Host**（`gemm_a8w8_blockscale_bpreshuffle_host.cc`）
   - 新增 `shuffle_b(src, dst, batch, N, K, block_k)`:OpenMP 并行,按上式重排。
   - main 里:`host_b_shuffled = shuffle_b(host_b)`,上传 `host_b_shuffled` 到 `dev_b`。
   - `kargs.stride_b = BLOCK_K;`（原为 `K`）。`stride_b_batch` 仍 = `N*K`（per-batch 总量不变）。
   - **CPU reference 仍吃原始 `host_b`**（`[N,K]`）,故任何 shuffle/寻址错误直接暴露为 FAIL。

2. **Kernel**（`gemm_a8w8_blockscale_bpreshuffle_kernel_template.hpp`,`gb_offset` lambda）
   ```cpp
   // 原基线: return half_tile_n * HALF_B_N * stride_b + tile_k * B_K;
   // 改为:
   auto gb_offset = [&](int half_tile_n, int tile_k) {
       return half_tile_n * T::HALF_B_N * kargs.stride_b + tile_k * kargs.n * T::B_K;
   };
   ```
   即 K 步进从 `tile_k*B_K` 改为 `tile_k*N*B_K`(跳一整个 K-panel)。
   base 指针 `col*stride_b = col*BLOCK_K` 自动指向 panel0 的第 col 行;N 维 layout 的
   `n_local*stride_b` 在 panel 内按行寻址;每行 BLOCK_K 个 ki 连续 → `VEC_B=16` 连续读成立。

### 关键推导结论（决定了为何改动如此之小）
- `make_layout_gb` 的 N 维分解 `n_local = 64*rep + 32*wn + 4*(lane/8) + wm`,是到 `[0,127]`
  的**恒等双射**。所以 B 打成 panel 后按行寻址无需任何 fragment reshuffle → preshuffle
  退化为**纯 stride/offset 变化**。
- **SFB 无需重排**:scale 按 `(n_group, k_group)` 寻址,与 B 的物理字节布局无关。`load_sfb`
  与 `sfb_ptr` 一律不动。

### 验证与性能（本机实测）
- 数值:`-v 1` 在多 shape（含 K>BLOCK_K 多 panel、多 batch）全部 **ALL BATCHES VALID**。
  已测:256x512x256(b2)、1024³、2048³、512x768x512(b3)。
- 性能（同机对照基线,batch=1,M=N=K）:

  | M=N=K | base TFlops | bpreshuffle TFlops |
  |------:|------------:|-------------------:|
  | 1024  |   79.5      |   79.4  |
  | 2048  |  291.2      |  296.0  |
  | 4096  | 1310.3      | 1315.1  |
  | 8192  | 1069.5      | 1108.0  |

  结论:与基线持平,大 shape 略优。收益来自 K-panel 内**跨 N 行的 cache line 打包**,
  而非去转置(B 在 `[N,K]` 下每行 K 本来就连续)。

### 资源占用（-Rpass-analysis,gfx950)
VGPR 256 / SGPR 73 / Occupancy 2 / LDS 135168 bytes / VGPR spill 24 / scratch 60B。
与基线基本一致(preshuffle 未增加寄存器/LDS 压力)。

## 2. 已同步更新的外围文件
- `README.md`(本目录):加了 "B preshuffle layout" 节、性能对照表、Files 更名。
- `.claude/skills/opus-kernel-authoring/SKILL.md`:新增 "Preshuffle / weight repacking" 节
  + checklist 一条,沉淀 5 条可复用经验(见下 §4)。
- 根 `README.md`:**未加行**——其 GEMM 表未收录任何 `opus_gemm/*` 子目录(连基线都没列),
  为保持一致性未单独加 bpreshuffle。

## 3. 后续方向:激进版(LDS-bypass,shuffle 成精确 MFMA fragment 序）

### 动机
最小版仍走 global→LDS→ds_read→寄存器 三跳。激进版设想:host 把 B 直接 shuffle 成
**MFMA operand 的 per-lane fragment 顺序**,kernel 里 global load 出来即为 MMA 输入寄存器,
**跳过 LDS 往返和 sb/rb swizzle**。这是 aiter `shuffle_weight(x, layout=(16,16))` 的路子。

### MFMA 16x16x128 fp8 的 B operand 布局(已推导,实现时据此写 shuffle）
- 指令:`__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4`(本 kernel 用普通非 scale 重载,
  scale 在寄存器手动乘)。
- B operand:每 lane 持有 `elem_b = W_N*W_K/warp_size = 16*128/64 = 32` 个 fp8。
  lane 映射:`n = lane % 16`,`k_group = lane / 16`(每 lane 覆盖 32 个连续 K)。
- aiter shuffle_weight 参考(`aiter/aiter/ops/shuffle.py:47`,layout=(16,16),fp8):
  ```
  IN,IK=16,16; BK=IK*2=32; K=16//1=16(fp8 一次连续粒度); BN=IN=16
  x.view(-1, N//16, 16, K//32, 2, 16).permute(0,1,3,4,2,5).contiguous()
  ```
  gfx1250 WMMA 变体见 `shuffle_weight_gfx1250`(同思想)。

### ⚠️ 关键权衡(为何"未必划算",必须先实测）
**本 kernel 的 B 在 LDS 里是被 4 个 M-wave（T_M=4）复用的**,不是单纯为了转置。
`make_layout_gb` 把 T_M 当 p_dim 编进 B 的 N 维 → 8 个 wave 协作载入整个 `HALF_B_N×B_K`
的 B tile 到 LDS,4 个 M-wave 都从 LDS 读同一份 B。
若 LDS-bypass,让每个 M-wave 各自从 global 直读自己的 B fragment,**B 的 global 带宽会 ×4**。
佐证:aiter 官方 flatmm pipeline(`opus_gemm_pipeline_a16w16_flatmm_gfx950.cuh`)即使 B 已
pre-transposed,**仍保留 LDS**(producer async_load→LDS,consumer ds_read→MMA)。

因此激进版**只在 B 复用度低时可能赢**(比如 T_M=1,或 M 很小),否则得不偿失。

### 实现前必做的实测(建议新会话第一步)
1. 用 rocprof / 现有 benchmark 测最小版 B 的 global 读字节数 & L2 命中率,估算 LDS 往返占比。
2. 若 B 的 ds_read/ds_write 不是瓶颈(很可能),则激进版收益有限,可不做或只在 T_M=1 变体做。
3. 若要做:新建 `block_scale_gemm_bpreshuffle_ldsbypass/`,host shuffle 改成 fragment 序
   (对照 shuffle_weight),kernel 删掉 B 的 sb/rb,`v_b` 直接来自 global load;
   A 侧保持不动;流水线 waitcnt/barrier 需重排(B 的 ds_read 消失,lgkmcnt 计数要改)。
   **风险高**:会打乱现有精细流水线,务必分步 + `-v 1` 逐 shape 验证。

## 4. 沉淀到 skill 的可复用经验(已写入 SKILL.md）
1. 从现有 global-load layout 出发,别重写——最小 stride/offset 改动优先。
2. 先推导 layout 的 N/M 分解是否恒等 → 决定是否需要 fragment reshuffle。
3. CPU reference 留在**原始 operand**,shuffle/寻址错误直接暴露。多 panel(K>BLOCK_K)必测。
4. scale 若按 group 坐标寻址,通常无需重排。
5. LDS-bypass 是独立且非平凡的决策:B 常因**跨 wave 复用**而留在 LDS,不只是转置;
   须实测再决定,别假设。
