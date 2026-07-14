# block_scale_gemm_ldsbypass 交接文档(激进版 LDS-bypass,**已完成**)

本目录是从 `block_scale_gemm_bpreshuffle/` 拷贝改造的**激进版**:让 B 绕过 LDS,
直接 global→寄存器 fragment 喂 MFMA。**实现与验证已全部完成**,结论如下,细节见 README.md。

## 结论(TL;DR)
- ✅ **正确**:repack kernel + 主 kernel + host 全部完成,多 shape `-v 1` 全 VALID
  (1024³、2048³、256×512×256 -b2、512×768×512 -b3、1024×1024×512、512×512×1024)。
- ✅ **LDS 确实降**:主 kernel LDS 135168 → **33792** 字节/block(4×),无寄存器 spill。
- ❌ **性能全面回退**(含小 M),符合原计划预测。**不作为 bpreshuffle 的替代**,仅作对照实验保留。
- ⚠️ **小 M 假设无法在本 tile 配置下验证**:`BLOCK_M=256`/`T_M=4` 固定 → 即便 M=256 仍有 4 个
  M-wave 共享 B,×T_M 放大照旧。要验证"小 M 占优"需专门做 **T_M=1 特化**(未来工作)。

## 环境(已核实可用)
- opus 头文件:`/root/workspace/aiter/csrc/include/opus/opus.hpp`
  编译前:`export OPUS_INCLUDE_DIR=/root/workspace/aiter/csrc/include`
- gfx950 在机,`/opt/rocm/bin/hipcc`(HIP 7.0)可就地编译运行。`make -j` 即可。
- 批准的完整计划:`/root/.claude/plans/abstract-puzzling-aho.md`

## 性能对照(batch=1,本机 gfx950,单位 TFlops)
| M=N=K | bpreshuffle | ldsbypass |
|---:|---:|---:|
| 1024 |   54.2 |  22.7 |
| 2048 |  156.0 | 163.3 |
| 4096 | 1156.9 | 696.4 |
| 8192 | 1019.9 | 761.4 |

小 M(n=k=4096/8192):256×4096×4096 68.2→38.3;256×8192×8192 144.3→84.5;512×4096×4096 136.9→90.8。

注:`M < 256` 在 **基线与本变体都会 GPU fault**(部分 tile 越界写 C),是既有限制,非本变体引入。

## 已实现的核心设计(把 v_b 当"不透明 blob",零手推)
两阶段 B 重排:
1. host `shuffle_b`:B[N,K] → K-panel B'(与 bpreshuffle 相同)。
2. **GPU 一次性 repack kernel**(`gemm_a8w8_blockscale_ldsbypass_repack_kernel`,template.hpp:219):
   复用主 kernel 完全相同的 `u_gb`/`u_sb`(载入 LDS)+ `u_rb`(读出 fragment),再把 fragment
   字节**原样连续**写入紧凑缓冲 B''。存/取都走同一个 `bpp_frag_offset`(template.hpp:207),
   fragment 中途不被解释 → MMA 输入与基线逐字节相同 → 正确性由构造保证,无手推风险。
   离线跑一次,不计入 benchmark(host.cc:292)。

B'' 布局:`B''[col_tile][k_tile][half_n(2)][wave_n(T_N)][lane(64)][FRAG_B_ELEMS(128 fp8)]`。
每 lane 128 fp8 连续,主 kernel 读 `b_frag_load_insts=8` 次连续 VEC_B load。B'' 总字节 = B 大小。

主 kernel(template.hpp:272):
- B 删掉 LDS 相关(smem_b/s_b/u_sb/u_rb 全不用),`load_bpp` lambda 直接 g_bpp→v_b 喂 MMA。
- A 侧完全不动(保留 LDS 转置 + 跨 N-wave 复用)。
- waitcnt:保守正确路线。单个 CDNA vmcnt 按发射序追 A 的 buffer_load_lds 再追 B 的 global load;
  每次用 B 前 `s_waitcnt_vmcnt(0)`;B 无需 barrier;A 保留 s_barrier。hn-outer 保证同时只活
  一个 B'' fragment(32 VGPR)→ 零 spill(v_c 本身已 128 VGPR)。

## 文件现状(全部完成,可直接 make)
- `gemm_a8w8_blockscale_ldsbypass_common.h`:traits 加 `FRAG_B_ELEMS=128`/`b_frag_load_insts=8`/
  `HALVES_N=2`;kargs 加 `ptr_b_pp`/`stride_b_pp_batch`。
- `..._kernel_template.hpp`:repack kernel(:219)+ 主 kernel(:272)+ `bpp_frag_offset`(:207)。
- `..._kernel.cc`:两个 kernel 的显式实例化。
- `..._host.cc`:host_b 原始布局喂 CPU ref;shuffle_b 生成 B';dev_b_pp 分配;
  计时外先 launch repack 填 B''(:292),再跑/计时主 gemm。
- `README.md`:已重写为 ldsbypass 版(动机、两阶段 repack、实测对照、"为何回退")。

## 方案 A 变体:host-pack(**已完成**,2026-07-09)
用户要求"连 B'' 的生成也不经过 LDS"。新增 `gemm_a8w8_blockscale_ldsbypass_hostpack_host.cc`:
删掉 GPU repack,改由 **CPU 生成 B''**,主 kernel 逐字节不变。`make hostpack` 或 `make -j` 构建。

- **零手推**:host 复用 opus layout。opus 求值链(make_layout/layout_to_offsets/coord_to_linear)
  本就 `__host__ __device__`;仅 `unfold_x_stride`/`unfold_p_coord`(opus.hpp:2703/2711)标 `__device__`,
  其体为纯 constexpr,新文件 namespace `hostlayout` 逐字复刻改 `__host__ __device__`,重建 u_gb/u_sb/u_rb。
- `host_pack_b()` 照抄 repack kernel:u_gb→CPU-LDS→u_rb→B''(bpp_frag_offset)。
- **关键坑(唯一不在 layout 里的东西)**:CDNA `buffer_load_lds` 把一个 wave 的 64 lane 散布到
  连续 LDS 槽——`u_sb` 只给 per-issue 的 **wave 基址(无 lane 项)**,硬件另加 `lane*VEC_B`。
  host 填 LDS 必须显式加 `lane_id*VEC_B`;`u_rb`(普通 ds_read)自带 lane 项,读出无需补。
  (漏这一步的症状:LDS 只填了 1/64,C 几乎全 0。)
- **铁证**:host-pack B'' vs GPU-repack B'' memcmp = **0 字节差异**(N=K=256);多 shape `-v 1` 全 VALID。
- opus 3-arg `make_layout<-1>` 有 bug(负 size),host 一律用默认 `make_layout`(layout_linear),
  offset 走 `layout_to_offsets<VEC_B>`。
- 性能:主 kernel 同一个 → gemm 吞吐不变;差别只是预处理从 GPU 挪到 CPU(慢但离线不计时)。

## 若要继续(可选后续工作)
1. **T_M=1 特化**:唯一能真正检验"小 M 占优"假设的路径。需新 traits(T_M=1、BLOCK_M 相应缩小)
   + 主循环去掉 A 的跨 M-wave 复用假设。工作量中等。
2. **收紧 waitcnt**:把保守 `s_waitcnt_vmcnt(0)` 换精确计数——但因瓶颈是 VALU 非访存,
   预期收益有限,当前 tile 下不值得投入。
3. 可选:向 `.claude/skills/opus-kernel-authoring/SKILL.md` 追加"GPU-repack 保证 fragment 一致 +
   LDS-bypass 前必须实测 LDS 是否瓶颈"的经验(见下)。

## 沉淀经验(建议写入 skill)
- **LDS-bypass 前必须实测 LDS 是否瓶颈**:本例 compute-bound(VALU 主导),B 的 ds_read 非瓶颈,
  去掉反而让 B 的 global 读 ×T_M 放大 → 全面回退。与 aiter flatmm"即使预转置也留 B 在 LDS"一致。
- **"把 fragment 当不透明 blob,存取用同一 layout"** 可零手推保证 preshuffle 正确性,
  代价是一个一次性 GPU repack kernel(离线权重预处理场景完全可接受)。
