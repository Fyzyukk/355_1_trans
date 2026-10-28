---
name: aac16-jobid
description: 使用 AAC16 已运行的 Slurm JobID 8438（Minimax 的 mi355-gpu-53 和 mi355-gpu-58）执行 CPU/GPU 任务、检查状态或运行 MORI 两节点 RDMA 测试。经 blyu_qle@aac16.amd.com 使用 srun --jobid，不重新 salloc/sbatch，不套用 Crusoe spur exec 或 GPU15 的 amd/root 登录规则。用户提到 AAC16、GPU53/58、job8438、现有两节点 allocation 或 MORI 跨节点通信时使用。
---

# AAC16 现有 JobID：GPU53 / GPU58

## 1. 第一件事：检查，不重新申请

用户已经授权并取得一个独占 hold 作业；**所有新任务应复用此 JobID**：

| 项目 | 已验证值（执行前仍需复查） |
|---|---|
| JobID | `8438` |
| 属主 / Account | `blyu_qle` / `bho` |
| Reservation | `aac16_mi355-gpu-53_mi355-gpu-58_reservation` |
| 节点 | `mi355-gpu-53` = `10.2.80.30`；`mi355-gpu-58` = `10.2.80.25` |
| 每节点 | 8× MI355X / gfx950 / 每卡约 288 GiB；256 逻辑 CPU；约 3 TB 内存 |
| 已分配总资源 | 16 GPU、512 CPU、6000000M 内存；`OverSubscribe=NO` |
| 原始到期 | 2026-11-19 09:02:39 北京时间（2026-11-19T01:02:39Z） |
| 提醒 | 2026-11-18 09:02:39 北京时间，需用户确认续期，不自动续期 |

查看 [占用记录](<../../../_docs/aac16-hold-8438.md>)，但不要把记录当实时状态。每次开始任务前确认：RUNNING、属主、account、reservation、NodeList、EndTime 全部正确；否则停止并询问用户。

**不要**再次 `salloc` 或 `sbatch`；不要 `scancel 8438` 来结束一个测试。一个 hold 作业同时持有两台节点，断开 SSH 不释放。

## 2. Windows 入口与便捷包装器

入口为 `blyu_qle@aac16.amd.com`，使用既有 key：

```text
C:/Users/blyu/Documents/Workspace/.ssh/id_ed25519_amdmail
```

不要输出、复制到跳板机、上传或提交私钥。工作区 [SSH 配置](<../../../.ssh/config>) 保留了其他登录条目；AAC16 包装器显式指定地址和用户名，不依赖 GPU15 别名。

[包装器 aac16-job.ps1](<../../../_scripts/aac16-job.ps1>) 已实测：

```powershell
# 当前工作目录为 C:/Users/blyu/Documents/Workspace 时：
.\_scripts\aac16-job.ps1 status 8438
.\_scripts\aac16-job.ps1 nodes 8438

# 53 / 58 / both，cpu 不分配 GPU 设备，gpu 分配每节点全部 8 GPU：
.\_scripts\aac16-job.ps1 cpu 8438 53 'hostname; id; pwd'
.\_scripts\aac16-job.ps1 cpu 8438 both 'df -hT /media/NVME "$HOME"'
.\_scripts\aac16-job.ps1 gpu 8438 both 'source /etc/profile.d/modules.sh; source /shared/apps/aac.modules.bash; module load rocm/7.0.2; rocm-smi --showuse --showmeminfo vram --showpids'
```

- `nodes` 和执行模式先在同一远端命令内验证作业身份/状态/account/reservation/节点范围、`OverSubscribe=NO`，并要求距EndTime至少10分钟；仍应人工先执行status确认预计任务能在剩余时间内完成。守卫不要求固定JobID=8438，允许经用户确认的同账户、同reservation、同节点继任作业；MORI运行器则明确pin当前8438。
- `cpu` = 每节点 2 CPU、`--gres=none`；`gpu` = 每节点 8 CPU、`--gres=gpu:8`；均为最多 10 分钟的 step。更大任务用第 3 节显式参数。
- 带空格、单双引号、分号的命令整体作为一个 PowerShell 字符串参数。包装器用 UTF-8/base64 传输，避免 Windows OpenSSH 多层引号丢失。
- Guard 和 exec 辅助脚本已经部署在本次共享目录，包装器依赖它们。分别对应 [job-guard.py](<../../../_scripts/aac16-job-guard.py>) 和 [job-exec.py](<../../../_scripts/aac16-job-exec.py>)；不要删除或覆盖供其他 session 使用的部署副本。
- 无重新申请、取消、自动续期功能。执行模式传入的是用户授权的任意Bash命令，**不是受限命令沙箱**；不要在payload里隐藏scancel/重提/系统修改，不得因包装器通过guard就把任意命令视为已授权。
- 包装器的status/nodes/cpu/gpu只读查询模式及含引号参数均已实测；JobID guard另有 [10项本地负向单元测试](<../../../_scripts/test-aac16-job-guard.py>)，包括错误属主、账户、节点、独占及过期拒绝。

不同 Windows 工作目录可用绝对路径调用包装器。默认配置内容不随工作目录改变，但 `.ssh/config` 相对路径会改变。

## 3. 在 AAC16 内使用 srun --jobid

人工登录控制节点：

```powershell
ssh -F C:/Users/blyu/Documents/Workspace/.ssh/config -i C:/Users/blyu/Documents/Workspace/.ssh/id_ed25519_amdmail -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile=C:/Users/blyu/Documents/Workspace/.ssh/known_hosts blyu_qle@aac16.amd.com
```

在控制节点上：

```bash
squeue -j 8438
TZ=UTC scontrol show job 8438
squeue --steps -j 8438

# CPU-only，两节点各执行一次：
srun --jobid=8438 --overlap -N2 -n2 --ntasks-per-node=1 \
  -c1 --gres=none --time=00:02:00 --kill-on-bad-exit=1 hostname

# GPU，单节点短时任务；命令路径应在共享目录：
srun --jobid=8438 --overlap -N1 -n1 -w mi355-gpu-53 \
  -c8 --gres=gpu:8 --time=00:10:00 --kill-on-bad-exit=1 \
  bash /shared/amdgpu/home/blyu_qle/<session>/task.sh

# 两节点各一个launcher，例如各启动8个local GPU worker：
srun --jobid=8438 --overlap -N2 -n2 --ntasks-per-node=1 \
  -c64 --gres=gpu:8 --time=00:05:00 --kill-on-bad-exit=1 \
  --output=/shared/amdgpu/home/blyu_qle/<session>/run-%N.log \
  bash /shared/amdgpu/home/blyu_qle/<session>/launcher.sh
```

`--overlap` 是为了允许步骤与已运行的 hold batch 同时存在，**不是 session 资源隔离**。其他 session 的 step 可以竞争同一批 CPU/GPU，必须串行或先约定节点/设备/时间。不要只看 `squeue` 作业行，应看 `squeue --steps -j 8438` 并检查实际进程和显存。

停止只应针对本 session 启动的精确步骤，如 `8438.<step>`；先验证步骤归属与用途，不动 `8438.batch` 或父作业。限时 step 结束不释放父 allocation。

人工需要 shell 可在确认状态后使用：

```bash
srun --jobid=8438 --overlap -N1 -n1 -w mi355-gpu-53 \
  -c8 --gres=gpu:8 --time=01:00:00 --pty bash -l
```

此交互形式未在非交互 agent 工具中实测。常规 agent 工作使用脚本 + 日志，不依赖交互终端。

## 4. ROCm、权限、存储：不要照搬 GPU15

两台节点系统 Python 没有预装 torch/mori/mpi4py，ROCm 默认不在 PATH。脚本中明确加载：

```bash
source /etc/profile.d/modules.sh
source /shared/apps/aac.modules.bash
module load rocm/7.0.2
```

实际工具目录：`/shared/apps/ubuntu/opt/rocm-7.0.2`。初始未加载环境产生过 `rocm-smi: command not found`；正确加载并分配 GPU 的 step 已能识别 8 张卡，不能把未加载工具误读成无 GPU。

- `blyu_qle` **无免密 sudo**，Docker socket 无访问权限；不要用 GPU15 的 `blyu + sudo` 配置套用这里。
- 用户态 Podman/Enroot 可用，但本次不需要容器，也未安装系统包。
- 当前用户 home：`/shared/amdgpu/home/blyu_qle`，共享 NFS，两节点均可访问；快照剩余约 33T。
- 本地大盘：`/media/NVME`，`/dev/md127`、XFS、56T；53 快照余约 55T，58 余约 54T；**不是 NFS，内容不跨节点共享**。
- 用户本地缓存根：`/media/NVME/var/tmp/blyu_qle`。53 的 `/data` 在根盘上，58 无 `/data`；不要凭 GPU15 经验往 `/data` 写大文件。
- 每个 session 建独立项目、日志、临时目录、端口和构建缓存；共享源码/构建树不要并发修改。

## 5. MORI 跨节点 RDMA 测试入口

本次测试独立目录：

```text
/shared/amdgpu/home/blyu_qle/aac-tests/mori-8438-20261020
```

- MORI 源码提交：`e6ddd2a73598911cf896c74b2c236cdf9cb11939`。
- 原生 MORI-IO binary：`mori-build/tests/cpp/bench_engine`，无需 PyTorch/Python mori。
- 已构建 `gfx950 + ionic`，启用对应 firmware 的 `IONIC_CCQE`；检测到真正 system pciutils header，未使用会禁用PCI拓扑的stub。
- 两台各 8 条 Pollara RoCE rail，端口报告 400 Gb/s；还存在管理 Broadcom NIC。**400 Gb/s 是标称链路速率，不是应用带宽测量。**
- 数据内存 `--mem-type gpu`、后端 `--backend rdma`，`MORI_DISABLE_AUTO_XGMI=1`；跨节点不是 XGMI/LSA。
- 控制面 `MORI_SOCKET_IFNAME=enp193s0f1np1`，IPv4 10.2.80.30/25；这不选择数据面 HCA。数据面使用该版本自动 PCI/rail 感知选择，必须检查实际网卡统计，勿盲设某个 GPU index→HCA。

[运行器](<../../../_scripts/aac16-mori-run-8438.sh>) 部署为 `run.sh`，检查 job/node + 每卡0%/低于1GiB显存/无KFD的即时门禁，默认最长150秒；不是独占证明，不能绕过门禁跑忙卡。它持有节点级协作flock，能排斥同运行器的另一session，但不能阻止忽略锁的任务。采集 admission、RDMA hw/net counters 及完成后显存；binary失败时也尽力收尾并返回原退出码，但srun的kill-on-bad-exit可能提前终止另一节点的收尾。缺失标准计数器目录不代表没有RDMA。

[串行复测脚本](<../../../_scripts/aac16-mori-repeat-8438.sh>) 部署为 `repeat.sh`，有全suite协作锁、自动unique run_id和可选 `AAC_MORI_PORT_BASE`，相同run_id已有日志会拒绝。它默认端口范围28500–28616，应确认无其他任务占用，必要时提供新的空闲基址。此次正式结果在补充协作锁前由Lead串行执行；不要声称锁已证明历史测试全程隔离。

复现单 GPU 对消息扫描（53 为 initiator，RDMA WRITE payload 53→58）：

```bash
ROOT=/shared/amdgpu/home/blyu_qle/aac-tests/mori-8438-20261020
# 先检查父job、其他步骤和实时GPU占用，选择独立端口及新日志名。
srun --jobid=8438 --overlap -N2 -n2 --ntasks-per-node=1 -c8 \
  --gres=gpu:8 --time=00:04:00 --kill-on-bad-exit=1 \
  --output="$ROOT/<unique-run>-%N.log" \
  bash "$ROOT/run.sh" mi355-gpu-53 --port <free-port> --op write --gpu 0 \
  --num-initiator-dev 1 --num-target-dev 1 \
  --all --sweep-start 1048576 --sweep-max 33554432 \
  --transfer-batch-size 1 --num-qp-per-transfer 8 --num-worker-threads 1 \
  --disable-chunking --iters 200 --warmup-iters 20 --log-level info
```

多 GPU 对并发使用 `--num-initiator-dev 8 --num-target-dev 8`，需要更大CPU step budget如 `-c64`。Rank0/1表示两端角色，binary自己fork每端GPU worker，**不要再用 srun -n16 重复启动这个多GPU binary**。反方向将首参数改成 `mi355-gpu-58`（WRITE payload58→53）。READ则initiator从target读取，payload方向相反；报告时写清。

**计量定义**（本次固定源码）：每GPU对 `bytes=message_size × batch × iters`，除以整个timed-loop秒数及1e9，单位十进制GB/s；warmup不计时，包含post+completion polling，stop-and-wait。`AGGREGATE`是各pair GB/s相加，**不是统一整机wall timer推导，也不是双向带宽**。不能将单对乘8当实测汇总。保持checksum校验，不使用 `--skip-validate`；message sweep只校验最后一个size的全部slots，而不是每个中间size。

2026-10-20 已完成短时RDMA WRITE测试（32MiB消息、batch4、4QP、4worker、每pair100次timed+10次warmup）：

| Payload方向 | GPU pairs | 3次pair带宽之和中位数 | 3次范围 |
|---|---:|---:|---:|
| 53→58 | 8 | 385.72 GB/s | 385.51–386.07 GB/s |
| 58→53 | 8 | 385.82 GB/s | 385.64–386.12 GB/s |

6次主结果每次8对数据校验全部成功，8条rail每次各增加14,763,950,080 payload bytes；不是双向同时传输或统一wall-window测量。单对同参数对照47.87GB/s。数值是该配置短时快照，不作为后续任务的性能保证。

MORI 官方基准说明：[MORI-IO Benchmark](https://github.com/ROCm/mori/blob/e6ddd2a73598911cf896c74b2c236cdf9cb11939/docs/MORI-IO-BENCHMARK.md)。本次结果和限制见 [测试报告](<../../../_docs/aac16-mori-8438-report.md>)。

## 6. gemm+comm fusion 参考边界

参考 [fusion gf_comm.cpp:53–58](<../../../_wt/aiter-gemm-fusion/csrc/gemm_fusion/sp_qkv/host/gf_comm.cpp#L53-L58>)：强制 `worldSize==lsaSize`、`rank==lsaRank`、`myNodeStart==0`，只支持single-node LSA。已有 [SDMA A2A benchmark](<../../../_wt/opus-dist-fp8-sdma/opus_gemm_dist/opus_gemm_a2a_lsa/sdma_a2a_bench.cc>) 报的是GiB/s，不能拿来改-n16测试跨节点或当作这两台的网络结果。本次选MORI-IO RDMA是同框架不同层，**未测试GEMM fusion性能或MORI-EP all-to-all**。

## 6.1 节点内 XGMI 测量（64MiB / batch1）

2026-10-20在两节点各完成56个有序GPU对的MORI-IO XGMI single-process矩阵，另对6个代表方向各补2次重复，所有最终64MiB数据校验通过。GPU间拓扑均1跳XGMI。

| 节点 | 56对中位数 | 最低–最高 |
|---|---:|---:|
| GPU53 | 60.205 GB/s | 60.16–60.37 GB/s |
| GPU58 | 60.190 GB/s | 60.13–60.38 GB/s |

这是`--backend xgmi --xgmi-single-process --enable-sess`的单次复制应用路径，64MiB、batch1、200timed/20warmup，whole-loop含提交及HIP完成轮询，不是XGMI硬件峰值或节点聚合带宽。各pair串行，不能乘GPU数/peer数当实测aggregate。详细 [节点内报告及矩阵](<../../../_docs/aac16-intranode-8438/mori-io-p2p-64m-report.md>)。

另尝试fusion风格的CCO SDMA all-to-all，独立构建成功，但本环境在`ccoWindowRegister/hipMemSetAccess`阶段报invalid argument；纯节点内GDR_CAPABLE=0配置仍失败。**暂无有效SDMA all-to-all带宽结果**，不得用上述点对点或旧fusion其他机器数值代替。该失败路径不能当成功复用例，需单独解决ROCm/VMM兼容性。

## 6.2 节点内 XGMI 测量（32MiB / batch4）

另一个独立配置：同固定MORI-IO commit，`--backend xgmi --xgmi-single-process --op write --enable-sess --enable-batch-transfer --batch-contiguous`，32MiB消息、batch4、200timed/20warmup；每节点56个有方向卡对串行，另对0↔1、0↔4、4↔5各方向重复3次：

| 节点 | 56卡对中位数 | 最低–最高 |
|---|---:|---:|
| GPU53 | 60.76 GB/s | 60.70–60.85 GB/s |
| GPU58 | 60.74 GB/s | 60.69–60.87 GB/s |

112卡对及36次代表方向复测checksum通过。连续batch合并成128MiB HIP peer-copy，不能将它与64MiB/batch1参数混为同一实验；同样不是XGMI硬件峰值、并发整机带宽或GPU发起SDMA all-to-all。

详细 [32MiB/batch4报告](<../../../_docs/aac16-intranode-8438-report.md>)、[统计矩阵](<../../../_docs/aac16-intranode-8438-results.json>)、[运行器](<../../../_scripts/aac16-xgmi-8438.py>)。入口已部署为`intranode-launch.sh`；先核实job/GPU、无其他测试，使用新run-id和日志，在一个节点执行：

```bash
ROOT=/shared/amdgpu/home/blyu_qle/aac-tests/mori-8438-20261020
srun --jobid=8438 --overlap -N1 -n1 -w mi355-gpu-53 -c8 \
  --gres=gpu:8 --time=00:08:00 --output="$ROOT/<unique>-53.log" \
  bash "$ROOT/intranode-launch.sh" --mode matrix --run-id <unique-run-id>
```

`smoke`=0→1，`matrix`=56有方向卡对，`repeat`=6方向各3次。持有和RDMA运行器相同的节点协作锁，只协调遵守锁的任务。SDMA A2A的额外VMM对齐尝试也未通过注册，未作为带宽结果；不能从P2P数字外推A2A总量。

## 7. Hold 生命周期

不自动取消、重提或续期。结束本session计算只结束自己的step，保留父job和提醒。续期须用户确认且查看 [hold记录](<../../../_docs/aac16-hold-8438.md#L63>)：原固定 `sleep 2592000` 也会退出，只延长 TimeLimit 不足以续期保持。过期/失败时不沿用旧JobID假定资源仍属于自己。

## 8. Muse Spark Open 816B + ATOM 常驻服务（2026-10-20 起，节点 58）

用户要求在 AAC16 mi355-gpu-58 上用 ATOM 把 Meta Muse Spark Open（816B-A42B，Quark MXFP4）跑起来。**已跑通并保持常驻**，完整证据与复现步骤见 [Muse Spark 运行报告](<../../../_docs/muse-spark-aac16-run-report.md>)。

- **状态（2026-10-25 复核 + 接续 hold）**：旧 hold `8438` 已于 **2026-10-22T16:03:22Z 因 NODE FAILURE 被 Slurm 取消**（`hold-53-58-20261020/8438.err`），非到期/非人为；**新 hold 为 `8627`**（`blyu-hold-53-58-20261025`，2026-10-25T11:45:12Z 起，EndTime 2026-11-24T11:45:12Z，`Requeue=1`，两节点 `ALLOCATED+RESERVED`，guard 已校验通过）。**所有 `srun --jobid=` 与 `aac16-job.ps1` 的 JobID 参数都要用 `8627`**；也可不依赖 jobid，直接 `srun --reservation=aac16_mi355-gpu-53_mi355-gpu-58_reservation -A bho -p 256C8G1H_MI355X_Ubuntu22 -w mi355-gpu-58 -N1 -n1 ...`。记录见 [hold 8627](<../../../_docs/aac16-hold-8627.md>)。节点 58 本地盘资产（模型 474 GiB/65 shards、镜像 graphroot 48 GiB、cache）与 NFS 源码/脚本均完好；58 上另有其他 session（ochain）遗留的已退出容器，勿擅动。
- 会话目录（NFS，两节点可见）：`/shared/amdgpu/home/blyu_qle/muse-spark-20261020/`，内含 `ATOM/`（PR #2641 head `628b4a2`）、`scripts/`、`logs/`；模型与容器在节点 58 本地盘 `/media/NVME/var/tmp/blyu_qle/muse-spark-20261020/`。
- 服务现状：后台 srun step **`8438.80`**（`-c128 --gres=gpu:8 --time=28-00:00:00`，2026-10-20 07:50Z 起），容器 `atom-muse-srv`，**端口 8000**，集群内可用 `http://10.2.80.25:8000` 直连。`squeue --steps -j 8438` 查看；停止只 `scancel 8438.80`（勿动 `8438.batch`），或在节点内 `podman --root /media/NVMe/.../podman-root stop atom-muse-srv`。
- 启动/重启：登录节点 `setsid nohup srun --jobid=8438 --overlap -N1 -n1 -w mi355-gpu-58 -c128 --gres=gpu:8 --time=28-00:00:00 --kill-on-bad-exit=1 bash /shared/amdgpu/home/blyu_qle/muse-spark-20261020/scripts/start_atom_server.sh > .../logs/atom-server-long.srun.log 2>&1 &`；冷启动约 10–11 分钟（权重 390 s + 图捕获 68 s）。
- 关键约束：镜像 `rocm/atom-dev@sha256:566aa3dc…f687e` 自带 atom=PR base `53e4fb5`，**必须** `-v $SRC:/atom-src:ro -e PYTHONPATH=/atom-src/ATOM` 才能带上 PR 的 muse_spark；`--max-model-len 32768` 必填；HTTP 端口是 `--server-port 8000`（`--port 8006` 是引擎内部端口）；`reasoning_effort` 放在 `chat_template_kwargs` 内；无 reasoning/tool parser，content 含模板文本属预期；仅 bf16 KV；首跑不加 EP。
- 同目录还有：模型下载脚本、镜像拉取脚本、GPU/容器探针、smoke/bench/long-context 脚本、PR 代码生效性验证脚本；本地证据副本在 [_docs/muse-spark-aac16-evidence](<../../../_docs/muse-spark-aac16-evidence>)，分析文档 [PR 分析](<../../../_docs/muse-spark-pr-analysis.md>) 与 [容器环境](<../../../_docs/muse-spark-container-env.md>)。
- Day-0 对标性能 + GSM8K 已测（2026-10-20）：单流 68.5 tok/s、4 并发 206.6、32 并发 820.6、64 并发(1k/1k) 1776、128 并发 1384.8 output tok/s；GSM8K 5-shot strict 0.9431 / flexible 0.9424。详见 [Day-0 对标报告](<../../../_docs/muse-spark-day0-bench-gsm8k.md>)，脚本 `bench_day0_atom.sh`/`bench_req64_1024_1024.sh`/`gsm8k_atom.sh`，证据在 `_docs/muse-spark-aac16-evidence/bench2/`。
