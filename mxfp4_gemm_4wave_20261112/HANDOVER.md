# MXFP4 4-wave GEMM 优化交接（2026-11-12）

## 1. 口径

- 只看 **8192×8192×8192**、batch 1、随机数据（A/B 随机 fp4 字节，scale 随机 UE8M0 124–130）。
- tile 固定为 256×256×256（B_K=256，不改），4 个 wave（2×2），C 输出 bf16。
- 开发机上 GPU0–5 一直被别人占用，所有数字都是在 **GPU6** 上测的；GPU7 不稳定，不用。同一个程序在 GPU2 上比 GPU6 快约 5–6%（11-11 实测 229.7 vs 243.4 µs）。
- 计时：`-w 1000 -i 200`，几个版本交替跑多轮。**不要用 `-w 5 -i 20`**：那一段正好落在功耗控制器的降频区间，会慢 10% 左右。
- 测之前确认卡空闲（`rocm-smi --showuse`）。被占用时，主文件会测出 600–1400 µs 这类异常值。

## 2. 编译环境

`Makefile` 里有两个路径，换机器时用 make 变量覆盖：

```bash
make CLANG23_ROOT=/path/to/pin-toolchain OPUS_INCLUDE_DIR=/path/to/aiter-blockscale-mxfp8-nn-nt-tt/csrc/include
```

- `CLANG23_ROOT`：带 `amdgpu_pin_agpr` / `amdgpu_pin_vgpr` 的工具链（开发机上是 `/root/toolchains/yuyzhang512-amdgpu-pin-op-dst-build`）。普通 ROCm 编译不了。
- `OPUS_INCLUDE_DIR`：aiter `blockscale-mxfp8-nn-nt-tt` 分支的 `csrc/include`（里面有 opus.hpp）。
- host 程序的参数：`-m -n -k -b`（K 必须是 512 的倍数），`-v 1` 校验，`-w` 预热次数，`-i` 计时次数。环境变量 `MXFP4_SFA_K_PATTERN=1`（每个 K group 的 scale 不同，能查出 K group 错位）、`MXFP4_UNIT_SCALE=1`（scale 全为 1.0）。persistent 版本还可以用 `PERSIST_GRID=N` 指定 block 数。

## 3. 目录

| 路径 | 内容 |
|---|---|
| 根目录 `gemm_a4w4_mxfp4_4wave_*`、`Makefile` | **主文件（当前最好的稳定版）**：非 persistent，pin A/B/C + unroll 2 + ① 三处小修 |
| `candidate_m1/` … `candidate_m5/` | 4 个候选版本，每个目录自带 kernel/traits/host/Makefile，可以单独编译 |
| `candidate_*/compare.sh` | 一键对比：`./compare.sh <GPU> <轮数>`。先对主文件（上一级目录）和本候选各做一次 8192³ 校验，再交替计时 |
| `traces/` | 8192³ 的 ATT trace（`*.tar.gz` 解开后用 ROCprof Compute Viewer 打开 `ui_output_agent_*` 目录）、各指令统计 csv、对比表 |
| `tools/` | 测速、trace、分析脚本（见第 8 节） |

## 4. 版本一览

| 版本 | kernel md5 | 相对上一版的改动 | 8192³ 正确性 | GPU6 计时（µs） |
|---|---|---|---|---|
| 主文件 | 3565421… | — | 通过 | **243.2**（4.52P） |
| m1 | 0a149f2a… | 任务 3（scale 改为 global→VGPR→transpose→LDS，main loop 不再 transpose）+ 任务 4/5（persistent，最后一个 k-tile 计算时提前加载下一个 tile；LDS 按 stage 排布，C 只用 stage 1 分两批暂存；写回不再按 quadrant 等 `vmcnt(0)`） | 通过 | 245.9（波动 242–254） |
| m3 | b26a482b… | block 数改为通过 kernel 参数 `kargs.grid_blocks` 传入（不再用 `__builtin_amdgcn_grid_size_x()` 读 dispatch packet）；prologue 先发 A0/B0 的读再做 scale transpose | 通过 | 243.6 |
| m4 | 6b727ebe… | 写 panel 时的 4×4 字节 transpose 改用 DPP（`v_mov_b32_dpp quad_perm` + `v_perm`，借鉴 PR6696），去掉 `permlane` 和 `v_mov` 复制 | 通过 | 约 243.9（GPU 有干扰） |
| m5 | 1330908b… | A/B 的 LDS-DMA：每个 wave 的 4 条 load 写相邻的 LDS 块，用 MUBUF 立即数 offset（0/1056/2112/3168）区分，共用一个 m0；`make_layout_ga/sa/gb/sb` 改了顺序，新增 `async_load_imm` | 通过（另测了 K=512/1024/7168、多 batch、persistent 多 tile） | 约 243.4（GPU 有干扰） |

注意：
- m1–m5 都是 persistent 版本，**traits 和 host 也改了**（m3 起多了 `grid_blocks` 参数，host 按 CU 数 launch）。要替换主文件，必须三个文件一起换。
- m4/m5 的计时是在 GPU6 被间歇占用时测的，主文件那一轮也在 240.8–246.1 之间跳，只能说「和主文件持平」。**请在空闲的 GPU2 上重测**：`cd candidate_m5 && ./compare.sh 2 7`（m1/m3/m4 同理）。

## 5. 主要结论

### 5.1 功耗上限
- 8192³、随机数据下，所有版本在稳态都卡在 1400 W 的功耗上限，时钟约 1.63–1.66 GHz。全零数据则是 2.4 GHz、172 µs。
- 只跑 MFMA、不做访存的天花板（`tools/mfma_ceiling`）：随机数据 6.85P @1759 MHz。
- 因此「trace 里省下 cycle」不一定变成「省时间」：耗时约等于总能耗 ÷ 1400 W。

### 5.2 trace 逐项（每个 tile、每个 wave 的 cycle，见 `traces/items_compare.md`）
你在 trace 里指出的点，m1–m5 都消掉了：写回的 `vmcnt(0)` 等待（1442→8）、`s_endpgm` 等 store（1131→约 200）、block 之间的空档（663→0）、main loop VALU（9807→约 5600）、`permlane`（m4 起没有了）、m0/SALU（m5：每轮 32→8 次）。但时间转移到了 `ds_read` 排队和 `buffer_load` 发射卡顿上，总 cycle 和计时基本不变。

### 5.3 co-execution（`traces/coexec_compare.txt`，`tools/prof/coexec.py`）
- main loop 里 MFMA 单元忙的比例只有 **77–80%**；128 对 MFMA（同一个 C 的 kk=0/1）里只有 50–56 对做到了 32 cycle 内完成。
- 不是指令塞得太多（256 个 MFMA 间隔里只有 3 个夹了超过 3 条指令），也不是 MFMA 之间的依赖（依赖和不依赖的 MFMA 停顿一样）。
- 多出的时间主要来自 **`ds_read` 排队**和 **`buffer_load` 发射时被 TA 卡住**（每条 25–35 cycle）。两者都集中在 k-tile 的前半段：LDS 同时在接收 DMA 写入、又要供 A/B 读取；4 个 wave 的 load 同时挤在 TA 上。
- 如果做到完美的 co-execution，main loop 能少约 21% 的 cycle；但受功耗上限影响，实际收益会打折扣。

### 5.4 和 PR6696 的对比（`traces/trace_8192_pr6696_fp8.tar.gz`、`tools/pr6696/`）
- PR6696（fp8，tile 256×256×128）也是非 persistent。trace 里四种开销都有，大小和我们差不多：block 之间空档 612、`s_endpgm` 1108、写回 `vmcnt(0)` 1370、prologue load 4008。它每个 tile 有 64 个 k-tile（我们是 32 个），开销被摊薄了一半，所以看起来不明显。
- 按每个 k-tile 算，计算量和搬运量两者相同；**main loop 效率我们更高**（2625 vs 2789 cycle/k-tile，相对 MFMA 理想值 78% vs 73%）。PR6696 的 VALU 几乎为零，是因为它的 scale 是 1x128，比 MXFP4 的 1x32 少 4 倍，不需要每个 k-tile 做 transpose。
- 借鉴过来的只有一项：DPP 方式的 transpose（m4）。

## 6. 试过但没有效果的（都在 GPU6 上测过）

| 尝试 | 结果 |
|---|---|
| prologue 先读 scale 再读 A0/B0（任务 1） | 持平 |
| B_a barrier 挪到 C00 开头（任务 2） | 持平 |
| scale 每个 lane 直接读 4 行再用 v_perm 拼（不用 permlane） | LDS 多读 4 倍，持平；行序没排好时有 bank 冲突，慢到 278 µs |
| A/B load 均匀撒到 4 个 quadrant | TA 不堵了，但数据来不及落地，B_b 处等待，慢 2–9 µs |
| C 直接从寄存器写回（不经过 LDS） | 慢 12 µs |
| ② 只改 layout 去掉 m0 | 不够：opus 的 `async_load(layout)` 把偏移加在 LDS 指针上，每条 load 都要设 m0；必须用立即数 offset（m5） |

## 7. 下一步建议

1. 在空闲的卡上重测 m3/m4/m5（`compare.sh`），并重抓 m4/m5 的 trace：m4/m5 的 trace 里 `ds_read` 停顿比 m3 多（430→655/714），需要确认是 DPP 写 panel 的 LDS bank 冲突，还是抓 trace 时 GPU 被占用造成的干扰。
2. 改法 2：把 DMA 写入、`ds_read`、`buffer_load` 在一个 k-tile 里错开（现在都集中在前半段），在「TA 不堵」和「数据来得及落地」之间找平衡。这是目前 co-execution 剩下的主要空间。

## 8. 工具说明（`tools/`，脚本里的路径都是开发机路径，用前改一下）

| 文件 | 用途 |
|---|---|
| `bench6.sh name template.hpp` | 拷贝主目录的文件 + 指定 template 到 `/tmp/var_<name>` 编译，跑 2 个小尺寸校验。环境变量 `HOST=`、`TRAITS=` 可替换 host/traits |
| `cmp6.sh name1 name2 …` | 交替计时 8192³（`GPU=` 选卡，`ROUNDS=` 轮数） |
| `clk6.sh label cmd…` | 长时间运行，同时采样功耗和时钟 |
| `guarded6.sh cmd…` | 等 GPU6 连续空闲 20 秒再执行 |
| `prof/att3.sh`、`prof/att4.sh` | ATT trace（att4 用带 `grid_blocks` 的 traits，配合 `launch_g.cpp`）；`GRID=256` 用于 persistent 版本 |
| `prof/items.py traceA traceB …` | 逐项对比（第 5.2 节那张表） |
| `prof/coexec.py trace 16` | co-execution 分析（第 5.3 节） |
| `prof/summ.py`、`prof/regions.py` | 按指令类别、按 prologue/main loop/epilogue 汇总 |
| `pr6696/launch_6696.cpp` | 直接加载 PR6696 预编译的 .co 运行（参数结构和我们相同，二维 grid） |
| `mfma_ceiling/mfma_ceiling.cc` | 纯 MFMA 天花板测试 |
| `merge_sf.py` | 把任务 3 的 scale 方案套到 persistent 版本上的生成脚本（m1 由它生成） |
