# candidate_m6 —— 在 candidate_m5 基础上修正 main loop 的 ds_read 堵塞，并把写回下半推迟到下一个 tile

生成：2026-11-12，在另一台 MI355X（GPU2）上开发。只改了 kernel 模板；host、traits、kernel.cc、Makefile、compare.sh 与 m5 相同。
kernel md5：cad7a484…

## 相对 candidate_m5 的改动（共 3 处）

| 改动 | 位置 | 原因（trace 证据） |
|---|---|---|
| e1：C11 中写 scale panel 的 `ds_write_b128` 固定在最后一条 ds_read 之后（`sched_group_barrier(0x200)`），scale 的 `buffer_load` 紧随其后（0x020） | `sched_c11()` | m4 改用 DPP 后 transpose 变快，编译器把 ds_write 提前到第 22 条 MFMA，紧跟的 `ds_read_b128` 一次 stall 109–120 cycle（m3 的 ds_write 自然在末尾，没有这个问题） |
| e2a：C00 的 16 条 `ds_read_b128` 从「前 16 条 MFMA 各一条」改为铺在前 24 条 MFMA 上（每 3 条 MFMA 2 条读） | `sched_c00()` | m3 起 VALU/permlane 被去掉后，16 条读连续发出，后 8 条每条 stall 11–16；单独做 e1 时 stall 还会从 C11 转移到 C00，两处必须一起改 |
| h1：写回下半（16 条 store）不再在本 tile 发出：LDS 读进 `v_pend_c[16]`（64 VGPR），在下一个 tile 的 prologue C00 里每 2 条 MFMA 穿插一条 store 发出；最后一个 tile 在 persistent 循环结束后补发；第一个 tile 用 range=0 的 buffer（不产生访存） | 写回段、prologue C00（`sched_c00_store()`）、循环外 `store_pend_c` | 256 个 CU 同时写回，store 发射卡在全局写带宽上，下一 tile 的 k-tile 1 预取又排在 32 条 store 后面。推迟后：写回下半段消失（2086 → 0），预取 stall 3102 → 1958，prologue 的 vmcnt 等待 862 → 442 |

寄存器：VGPR 512 / AGPR 256，无 VGPR spill；SGPR spill 6（m5 为 4，溢出到 VGPR lane）。

## 结果（GPU2，8192³，`-w 1000 -i 200`，交替 7 轮中位数）

注意：这台机器的卡比开发机慢约 20%（纯 MFMA 天花板 5.65P vs 6.85P），只能看同卡的相对差。

| 版本 | µs | 相对主文件 |
|---|---|---|
| 主文件 | 286.9 | — |
| m5 | 286.6 | −0.1% |
| m5 + e1 + e2a | 281.7 | −1.8% |
| **m6（+ h1）** | **278.8** | **−2.8%** |

满载时钟 1400 MHz（m5 为 1437）：MFMA 更忙、功耗更高，trace 上省下的 cycle 约一半被功耗上限吃掉。

main loop（每趟 2 个 k-tile）：5123 → 4856 cycle，MFMA 忙碌比例 79.9% → 84.4%。

正确性：8192³（随机 scale）；1024×1024×512、1024×2048×1024×b2、2048×1024×7168×b3、768×1280×1536、256×256×512，各配 `MXFP4_SFA_K_PATTERN=1`，`PERSIST_GRID` 取默认 / 100 / 37 / 1，每种 2 次，全部 VALID。

## 试过但没有采用的（都在 GPU2 上和 m5/e1e2a 对比）

| 尝试 | 结果 | 原因 |
|---|---|---|
| e2b：C00 的 16 条读铺满 32 条 MFMA | 与 e2a 持平 | — |
| 下一 tile 的 scale panel 1 提前到 last tile 取 | 更差 | 编译器在新分支上插了更保守的 vmcnt |
| k-tile 1 预取放到写回下半的 store 之前 | 持平 | 等待只是换了位置 |
| 写回上下两半都推迟（32 条 store 进 prologue C00） | +1.8% | store stall 堵住 MFMA |
| 去掉 store 的 `nt` | +0.3% | — |
| C 清零挪到 prologue 的等待里 | 持平 | 编译器把常量 0 沉到 main loop 入口前生成，源码层面挪不动 |
| 去掉 B_a，16 条预取全放 C01 | +4.8% | TA 发射密度翻倍，stall 233–309 |
| barrier 前先发 4 / 8 条 MFMA | +0.5% / +1.7% | 这些 MFMA 依赖 C00 刚读入的数据，多出 lgkmcnt 等待 |
| CU 错峰：一半 block 开头 `s_sleep` 约 4k cycle（每个 XCD 内一半） | −0.35% | 收益依赖 block→CU 映射和整数 tile 数，没有放进 m6 |

## 剩余开销（每 tile 每 wave，trace）

- main loop：约 11000（ds_read / buffer_load 发射卡顿）+ barrier 约 2000（wave 之间的进度差）。
- tile 衔接：下一 tile 预取 stall 约 2000（所有 CU 同时预取）、写回上半 store 约 1400（全局写带宽）。
- 纯指令：C 转 bf16 + 暂存 LDS 约 2400、C 清零约 800、地址计算约 500。

## 测试

```bash
cd candidate_m6
OPUS_INCLUDE_DIR=<opus include> ./compare.sh 2 7
```
