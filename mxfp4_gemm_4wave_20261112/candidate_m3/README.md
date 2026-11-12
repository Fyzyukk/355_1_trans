# candidate_m3 —— 在 candidate_m1 基础上降低 prologue 的 lgkmcnt 等待

生成：2026-11-11，GPU6 上开发。主文件和 `../candidate_m1/` 都没有改动。

## 相对 candidate_m1 的改动

| 改动 | 原因 |
|---|---|
| block 数通过 kernel 参数传入（`kargs.grid_blocks`，traits 和 host 都改了），不再用 `__builtin_amdgcn_grid_size_x()` | 后者要读 dispatch packet（在 host 内存里），每个 wave 启动时等约 1000 cycle，而且是 m1 计时大幅波动（242–254 µs）的原因 |
| prologue：先过 barrier 发出 A0/B0 的 LDS 读，再做 scale panel 的 transpose 和写 LDS，然后第二个 barrier 后读 scale | transpose 盖住 A0/B0 的读延迟 |

## 结果（GPU6，8192³）

- 正确性：8192³ errors=0（随机 scale、`MXFP4_SFA_K_PATTERN=1`），无寄存器溢出。
- 时间（`-w 1000 -i 200`，与主文件交替 7 轮）：主文件 243.5，**m3 243.6（持平）**，m2（只改了 grid 参数）244.3，m1 245.9。

trace 逐项（`trace_8192_m3/items_vs_main_and_m1.md`，每个 tile 每个 wave 的 cycle）：prologue lgkmcnt 2380 → 813（其中约 475 是每个 wave 启动时读 kernel 参数，只发生一次）；合计 103.5k → 99.3k（主文件 100.9k）。

trace 目录：`trace_8192_m3/att/ui_output_agent_15421_dispatch_1/`。

## 在 GPU2 上测试

```bash
cd /root/workspace/mxfp4_gemm_4wave/candidate_m3
./compare.sh 2 7
```
