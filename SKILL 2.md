---
name: aiter-gemm-config-review
description: Review ROCm/aiter GEMM config PRs（model_configs/*_tuned_gemm.csv + *_untuned_gemm.csv 的更新）against the 4 standard checkpoints — ① 新增行不得用 hipblaslt/blaslt（只能 torch native / asm / flydsl / triton / opus，且不应新增纯 torch 行）② M 值纪律（prefill M>256 必须是 2 的幂、靠 getPaddedM 覆盖任意 M；decode M≤256 允许 CUDA-graph 精确值；防 shape 行数膨胀）③ 新旧 kernel 性能无回退（CSV 数据自洽 + 作者 A/B 证据：服务范围内总时间赢 main、单点不低于 ~0.93x）④ tuned 新增 shape 必须同步进 untuned（新增或已有，两文件均无重复行）。Use whenever asked to review/检查 aiter 的 config/tuning PR、GEMM CSV diff、"config update" 类 PR，或需要核对 tuned/untuned 同步。
---

# aiter GEMM config PR review（4 检查点）

适用于只改 `aiter/configs/model_configs/*_tuned_gemm.csv` / `*_untuned_gemm.csv` 的 config PR。参考实例：PR #6411（Kimi-K3 gfx950 BF16 GEMM tuned configs）。

## 输入：4 个文件

PR 分支与 main 分支各自的 tuned + untuned CSV。获取方式：

- **DSH 沙箱内 pwsh/curl 无网络**（schannel/TLS 被拦），用 fetch 工具拿 raw 文件：
  - `https://raw.githubusercontent.com/<pr-owner>/<repo>/<pr-branch>/aiter/configs/model_configs/<name>.csv`
  - `https://raw.githubusercontent.com/<org>/<repo>/main/aiter/configs/model_configs/<name>.csv`
  - 大文件（tuned 常 >100KB）fetch 会 spill 到临时目录（结果末尾给路径），copy 进 workspace 即可；spill/包装头脚本自动跳过。
- 有 gh/git 的环境：`gh pr diff <N>` 或直接 `git fetch origin pull/<N>/head` 后取文件。

## CSV 格式（实测）

- tuned: `gfx,cu_num,M,N,K,bias,dtype,outdtype,scaleAB,bpreshuffle,libtype,solidx,splitK,us,kernelName,err_ratio,tflops,bw`（18 列）
- untuned: `M,N,K,bias,dtype,outdtype,scaleAB,bpreshuffle`（8 列）
- dispatch key = tuned 前 10 列；**同 (M,N,K) 不同 gfx（如 gfx1250 vs gfx950）不算冲突**。

## 4 个检查点与通过标准

### ① blaslt 禁用
- 新增行 `libtype` 不得含 `hipblaslt`/`blaslt`；也不应新增 `torch` 行（与 main 默认路径相同，无意义，应删）。
- 文件中**存量** hipblaslt 行（如 outdtype=float32 的旧行）单独列出说明，不算本 PR 的问题。

### ② M 值纪律（odd M / shape 膨胀）
- prefill 行（M>256）**必须是 2 的幂**（512/1024/2048/.../16384），任意 M 靠 `getPaddedM` pad 到这些行；出现非 2 幂 M>256 = **FAIL**。
- decode 行（M≤256）允许非 2 幂精确值（CUDA-graph batch size，如 24/48/56/72/80/96），正常。
- 行数爆炸（为每个 profiled M 建精确行、动辄数百行）= **FAIL**；正确做法是 2 的幂 bucket + padding。
- 例外：固定 chunked-prefill 大小的精确行（如 M=16384）可接受，但要求作者说明理由。

### ③ 新旧 kernel 性能无回退
- **脚本可独立验证（数据自洽）**：`tflops = 2·M·N·K/us`、`bw = 2·(M·K+N·K+M·N)/us` 与 CSV 列误差 <2%（CSV 两位小数舍入）；flydsl 行 `splitK` 列 == kernel 名 `ksN`；不一致说明行是手搓/拼错的。
- **作者 A/B 证据（不可独立验证，需审视）**：每个新行在其服务的 M 范围内总时间必须赢 main，单点最差不得低于 ~0.93x；`total≈1.00x` 的边缘行（零收益但有单点回退风险）要逐个点名让作者解释去留；PR 文本与 CSV 数值不一致（如同一行耗时两处不同）要指出。
- `err_ratio>0` 的行（split-K 固有，积累顺序改变）→ 精度提示：要求端到端精度验证，或放在独立可 drop 的 commit。

### ④ tuned ↔ untuned 同步
- 每个新 tuned (M,N,K) 必须出现在 untuned 文件中（新增或 main 上已有）。
- 两个文件各自无重复行；新增 untuned 集合 == 新增 tuned shape − 已有 shape。

## 通用检查（额外发现里报告）

- **rebase 时效**：main 有而 PR 分支没有的行 = main 在 PR 分支后新增的内容；纯新增合并无冲突，但提醒 rebase。
- **跨 arch 同 shape**：同 (M,N,K) 不同 gfx 正常；同 gfx 同 10 列 key 冲突 = FAIL。
- 跑 `op_tests.tuning_tests.test_config_shape_collision`（作者应已跑，26 tests OK 之类）。

## 用法

把 4 个文件准备好后：

```powershell
python .claude/skills/aiter-gemm-config-review/scripts/analyze_gemm_config.py `
  --pr-tuned <pr_tuned.csv 或 spill.txt> --main-tuned <main_tuned.csv 或 spill.txt> `
  --pr-untuned <pr_untuned.csv> --main-untuned <main_untuned.csv>
```

输出按 4 检查点给 PASS/FAIL + 证据；任一 FAIL 时 exit code = 1。

## 报告模板

按此组织 review 结论（实例见 PR #6411）：

1. 检查点 ①②③④ 各给 ✅/⚠️/❌ + 关键证据（数字、shape、行数）。
2. 额外发现：rebase 需要、跨 arch 参照、split-K 精度提示、文本/CSV 不一致。
3. 结论与建议（approve / 要求修改 + 具体待办）。
