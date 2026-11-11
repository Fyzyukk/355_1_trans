# 第三章重点：MI300 Attainable Bandwidth

原文：[AMD Instinct MI300 GPU Memory Bandwidth](./mi300_gpu_bandwidth.pdf)，第 3 章，PDF 第 6–19 页。

本章回答的是：**硬件有理论带宽上限，程序实际能够持续提供多少请求，哪些环节使请求排队、使 wave 等待？**

下文按原文解释公式和数值。文中的 FIFO 深度、仲裁效率、64 KB LDS 等参数属于这份 2024 年文档的模型；用于当前 MI355X 时，需要核对对应硬件参数。

## 3 开头：用 N = λT 解释为什么需要隐藏内存延迟

### 核心关系

```text
N = λ × T
λ = N / T
```

| 量 | 含义 | 如果把分析对象定为内存请求 |
| --- | --- | --- |
| `N` | 系统平均在途数量 | 已发出、仍在系统中等待完成的平均请求数 |
| `λ` | 吞吐率 | 每 cycle 或每秒完成的请求数 |
| `T` | 平均停留时间 | 请求从进入到离开系统的平均时间 |

如果每个请求平均传输 `R` 字节，则可以进一步写成：

```text
带宽 BW = λ × R = N × R / T
目标在途请求数 N_required = BW_target × T / R
```

后两式是对 Little 定律的字节带宽换算。请求大小不同时，应采用一致的平均值和统计范围。

**本章的起点：要提高吞吐量，需要维持足够多的有效在途请求，并减少请求或 wave 的等待时间。** `N` 不等于单独的 wave occupancy：即使驻留很多 wave，如果它们都在等待依赖、没有继续发访存指令，内存管线仍可能供给不足。Little 定律中的 `N`、`λ`、`T` 必须统计同一类对象；计算内存请求吞吐时应使用请求停留时间，分析 wave 吞吐时才使用 wave 寿命。

### 开头列出的六类限制

| 原文列项 | 限制什么 | 对 N 和 T 的影响 |
| --- | --- | --- |
| **Software design and optimization**：软件设计与优化 | 工作组大小、寄存器/LDS 使用量、访问宽度、地址排列、计算与访存重叠 | 决定能驻留多少 wave、每 wave 能提供多少有效请求，以及需要等待多久 |
| **Frontend workgroup and wavefront dispatch**：前端 WG/wave 派发 | CPC 是否及时提供工作组，SPI 是否及时把 wave 分给 CU | 供给不足导致 wave slot 空闲，降低有效并发，并增加派发间隙 |
| **Compute Unit instruction scheduling and issuing**：CU 指令调度与发射 | 各 wave 争用共享 VMEM，指令依赖和其他管线等待 | 限制请求发出速率，拉长 wave 寿命 |
| **Vector L1 cache**：向量 L1 | TA 地址处理、缓存策略、tagram 分布、地址转换、FIFO 和 L2 接口 | 限制每 cycle 的处理量、可容纳的在途请求，并向 CU 回压 |
| **Vector L2 cache**：向量 L2 | 通道仲裁、缓存策略、队列、SoC 出口 | 限制所有客户端的汇总吞吐，并把下游等待传回 L1 |
| **SoC bandwidth-latency dynamics**：SoC 带宽与延迟变化 | fabric、LLC、内存控制器、HBM 的服务能力 | 决定请求完成速度和往返延迟；拥塞会向 L2 回压 |

原文把 SoC 侧视作黑箱，主要分析计算侧到 L2 的行为，并假定频率固定。随着优化推进，瓶颈可能从一个环节转移到另一个环节。

## 3.1 Frontend Workgroup and Wave Dispatch：前端能否持续供给工作

这一节分两次比较：**chunk 的供给与消耗；wave 的执行寿命与 slot 的补充周期。**

### 3.1.1 Workgroup Dispatch：CPC 再拿到 chunk 前，SPI 会不会已经派发完

先明确符号：

| 符号 | 含义 |
| --- | --- |
| `C` | 一个 chunk 里有多少个工作组 |
| `WG` | 每个工作组有多少个 wave；这里单位是 wave/WG |
| `#XCD` | 系统的 XCD 数量 |
| `#SPI` | 每个 XCD 的 SPI 数量，本文为 4 |
| `C_p` | 某个 CPC 轮转到下一个属于自己的 chunk 的周期 |
| `C_s` | 一个 SPI 派发完当前 chunk 分给自己的全部 wave 所需时间 |

**① CPC 再获得 chunk 的周期。** CPC 按轮转方式领取 chunk，遍历速度为 1 工作组/cycle：

```text
C_p = #XCD × C                         [cycles]
```

例如 8 个 XCD，每个 CPC 领取间隔为 8 个 chunk。

**② SPI 消耗自己那份工作所需时间。** 每个 SPI 分到 `C / #SPI` 个工作组，每组 `WG` 个 wave，SPI 派发速度为 4 cycles/wave：

```text
C_s = (C / #SPI) × WG × 4              [cycles]
```

这里的“消耗”指把 wave 派发到 CU，不是这些 wave 已经执行完。

**③ 比较两者，判断工作组派发限制。**

```text
C_s < C_p
→ SPI 已派发完当前工作，但下一份 chunk 还没到
→ 前端工作供给不足

(C / #SPI) × WG × 4 < #XCD × C
WG < #XCD × #SPI / 4

本文 #SPI = 4，因此：WG < #XCD
```

对 8-XCD 的 MI300X，本文得到 `WG < 8 wave/工作组` 的界限；对 4-XCD 配置，相同推导得到 `WG < 4`。

例如 `#XCD = 8、C = 8、WG = 4`：

```text
C_p = 8 × 8 = 64 cycles
C_s = (8 / 4) × 4 × 4 = 32 cycles
```

SPI 32 cycles 就能派发完自己分到的 wave，而 chunk 供给周期为 64 cycles。按照这个简化模型，前端消耗工作快于工作到来。两者之比为：

```text
C_s / C_p = WG / #XCD                  [本文 #SPI = 4]
```

这个比值是从原文两式得到的辅助判断量。小于 1 表示供给存在间隙；它不能直接当作实际 kernel 的利用率，因为资源不足、回压、派发重叠等还会影响运行。

**这一小节给的信息：工作组太小时，多 XCD 轮转派发可能使 SPI 缺少工作。判断要同时使用工作组的 wave 数和 XCD 数。**

### 3.1.2 Wave Dispatch：wave 结束后，slot 能否及时补上

SPI 派发工作组前，必须确认该工作组所有 wave 的 VGPR、SGPR、LDS、barrier、wave slot 等资源都可满足。因此先有一个**静态资源允许的驻留上限**。

例如本文的 CU 有 64 KB LDS，工作组使用 32 KB LDS、包含 4 个 wave：

```text
最多驻留工作组数 = 64 KB / 32 KB = 2
由 LDS 决定的驻留上限 = 2 × 4 = 8 waves/CU
```

其他资源还可能进一步收紧这个上限。这里的 64 KB 是原文例子的硬件值。

接着，原文从 wave slot 的角度计算轮转补充周期。设：

- `W_o`：本小节按静态资源约束采用的每 CU 目标驻留 wave 数。
- `T_p`：工作分配轮转一圈，使同一 wave slot 获得下一 wave 的派发周期。
- `T_s`：一个 wave 从建立到退出的执行寿命。

原文公式 (2)–(3)：

```text
T_p = C × #XCD × #SE/XCD × #CU/SE × W_o / (WG × C)
    = #TotalCUs × W_o / WG                          [cycles]
```

`C` 在推导中消掉了；这个周期取决于总 CU 数、目标驻留 wave 数以及每工作组 wave 数，并沿用原文的轮转派发假设。

判断条件是：

```text
T_s < T_p → wave 结束速度快于 slot 补充速度 → wave dispatch bound
```

你说的“wave 寿命与 slot 重新装填所需时间的比值”可以整理成：

```text
ρ_wave = T_s / T_p
```

**这是方便阅读而定义的比值，原文直接给的是 `T_s < T_p` 条件。** `T_p` 表示同一 slot 获得 wave 的轮转周期，并不是 slot 空下来之后才开始计时的纯空闲时间。

| 比值 | 按原文模型怎么判断 |
| --- | --- |
| `ρ_wave < 1` | wave 提前完成，slot 存在等待补充的间隙，可能受 wave 派发限制 |
| `ρ_wave = 1` | wave 寿命与补充周期相当 |
| `ρ_wave > 1` | 寿命足以覆盖这个补充周期；仍需检查执行、资源和访存限制 |

原文例子：MI300X 有 304 CU，`WG = 4、W_o = 32`：

```text
T_p = 304 × 32 / 4 = 2432 cycles
T_s < 2432 cycles 时，本文模型判断为 wave 派发受限
```

如果 `T_s = 1200 cycles`，则 `ρ_wave ≈ 0.493`，说明 wave 在该轮转补充周期的一半左右就已结束。**这个比值不是全 GPU 的实测 occupancy，也不意味着应主动拖长 wave。优化目标是提高有效供给和吞吐量。**

**3.1 合起来给出的信息：** `C_s / C_p` 检查 chunk 层面的工作供给，`T_s / T_p` 检查 wave slot 层面的持续供给；静态资源允许驻留多少 wave，还需要结合这些派发时间判断实际能否持续达到该并发。

## 3.2 Compute Unit：把 wave 的工作量和时间换算成每 CU 带宽

### 硬件背景

一个 CU 的 4 个 SIMD 共用一条 VMEM 管线。CU 向向量 L1 传命令、地址和写数据，三者分别有 FIFO，深度为 48、48、64。

一个 wave 有 64 个线程；VMEM 分 4 个阶段，每阶段处理 16 个线程。因此无下游回压时，一条指令信息约 4 cycles 传完。TA/L1 接收不及，会回压 VMEM，增加指令间距，拉长 wave 寿命。

### 原文公式 (4)–(5)

```text
BW_CU = W_o × n × R / (T_d + T_s)                  [B/cycle]

BW_CU = min{W_o × n × R / (T_d + T_s), 64}         [B/cycle]
```

| 量 | 原文定义 | 公式中提供的信息 |
| --- | --- | --- |
| `W_o` | 稳态每 CU 平均活跃 wave 数 | 多少 wave 在共同提供请求 |
| `n` | 每 wave 的内存指令数 | 每个 wave 提供多少次访问 |
| `R` | 每条内存指令的请求字节数 | 每次访问传多少字节 |
| `T_d` | 前一个 wave 结束到下一个 wave 开始之间的间隙 | 前端补充工作造成的等待 |
| `T_s` | wave 建立到退出的执行寿命 | 包括计算、依赖、访存、barrier 和管线等待 |
| `64` | CU ↔ 向量 L1 的 64 B/cycle 通路上限 | 请求供给再充足，也受该通路限制 |

64 个线程各读 8 B 的 `buffer_load_dwordx2`：

```text
R = 64 threads × 8 B/thread = 512 B/instruction
n × R = 一个 wave 请求的总字节数
```

当访问宽度不一致时，`n × R` 应理解为逐条请求字节数的总和，或使用相应的平均请求大小。

**公式的含义：用参与供给的 wave 数乘每 wave 请求量，再除以包含派发间隙的周转时间，估算每 CU 每 cycle 能提供的字节数，最后取 64 B/cycle 上限。**

达到这个上限需要在模型中满足：

```text
W_o × n × R ≥ 64 × (T_d + T_s)
```

### 由公式得到的优化方向

- **提高有效并发 `W_o`：** 减少不必要的 VGPR、LDS 等资源占用，使更多 wave 能提供请求。
- **提高每 wave 的有效请求量 `n × R`：** 选择合适的访问宽度和工作量；额外无用访问也会增加计数字节，不能视作有用带宽提升。
- **减少 `T_d`：** 改善 3.1 所讨论的工作组/wave 供给。
- **减少 `T_s` 中的等待：** 重叠计算与访存，减少依赖、barrier、缓存和下游排队。

这些量会互相影响。例如增加驻留 wave 会提高并发，也可能使共享 VMEM 更拥挤，导致 `T_s` 增长；增加请求量也可能填满下游 FIFO。因此不能假定只改分子、分母保持不变。

**符号使用要留意：** 3.1 的 `W_o` 用于静态资源允许的目标驻留量，3.2 则把它描述为实际稳态平均活跃 wave 数。原文同时用 `T_d + T_s` 表示 slot 周转时间，属于简化估算；若已有按完整时间窗测得的平均活跃 wave 数，再额外扣除同一派发间隙，可能重复计算空闲影响。实际代入时应统一 occupancy 和时间的统计口径。

**这一节给的信息：每 CU 带宽由有效并发、每 wave 请求量、前端间隙和执行寿命共同决定；3.3–3.5 接下来解释请求速率和等待时间受哪些硬件环节约束。** 这里的字节数是 CU 侧请求量；访问命中缓存时，它与物理 HBM 流量会不同。

## 3.3 Texture Addressing Unit（TA）：每条指令的地址要处理多久，折合多少 B/cycle

TA 位于 CU 和 TCP 之间，把 VMEM 指令转换为 TCP 可以查找的地址信息；返回方向由 TD 重组数据并交给 CU。

### 计算①：地址传递时间

- CU 理想输入：4 cycles/指令。
- TA 输出：每 cycle 向 TCP 传 4 个地址。
- 未合并：64 个线程地址需要 `64 / 4 = 16 cycles`。
- 可合并的最佳情况：每个地址编码最多 4 个连续线程访问，64 个线程变成 16 个地址，需要 `16 / 4 = 4 cycles`。

TA 的这种地址合并最多支持每线程 1 DWORD（4 B）。本文中的 `dwordx2/x4` 无法使用这种 TA 合并。

### 计算②：每条指令的字节吞吐

根据原文表 5，可以用下面的辅助计算理解：

```text
TA 环节的字节吞吐上限 = 每条 wave 指令请求字节数 / TA 处理该指令地址所需 cycles
```

| 指令 | TA 合并 | 每线程字节数 | 每 wave 指令字节数 | TA 处理时间 | 该环节每 CU 上限 |
| --- | --- | ---: | ---: | ---: | ---: |
| `buffer_load_dword` | 否 | 4 B | 256 B | 16 cycles | 16 B/cycle |
| `buffer_load_dword` | 是，最佳情况 | 4 B | 256 B | 4 cycles | 64 B/cycle |
| `buffer_load_dwordx2` | 否 | 8 B | 512 B | 16 cycles | 32 B/cycle |
| `buffer_load_dwordx4` | 否 | 16 B | 1024 B | 16 cycles | 64 B/cycle |

### 得到的限制与结论

未合并指令以 4 cycles/条进入 TA，却需要 16 cycles/条才能向 TCP 传完地址。连续突发会积压；原文展示约 12 条指令的突发就能导致 FIFO 溢出压力和 CU 回压，发射间隔随后拉长到约 16 cycles/条。

**3.3 算的是地址处理速率对应的每 CU 字节吞吐。** `dwordx2` 每 wave 请求 512 B，但 TA 要用 16 cycles，所以这一环节只能提供 32 B/cycle。`dwordx4` 虽然也无法 TA 合并，但每指令传 1024 B，能够对应 64 B/cycle，并减少完成同样字节量所需的指令及部分 tagram 查找压力。

原文因此建议在访问需求允许时使用 `dwordx4`。表中的“达到每 CU 上限的 50%”描述的是这一环节，不能直接换成“整机 HBM 利用率 50%”；整机还取决于活跃 CU 数、缓存命中及后续路径。

## 3.4 TCP：L1 的处理速率、可用并发容量和排队限制

TCP 就是向量 L1 缓存及控制器。**这一节没有给一个新的总带宽公式，而是逐部件说明：请求能否并行处理、最多缓存多少等待中的请求、哪里会回压。**

基本组织：4 个 tagram，每个 64 条缓存行，每条 128 B：

```text
L1 总容量 = 4 × 64 × 128 B = 32 KB
```

### 3.4.1 缓存策略：访问能否复用 L1 中的数据

| 访问 | Temporal | Non-temporal |
| --- | --- | --- |
| Read，wave/group scope | 允许命中，LRU 更新 | Forced-miss，Evict 更新 |
| Read，device/system scope | Forced-miss，Evict 更新 | Forced-miss，Evict 更新 |
| Write | Forced-miss，LRU 更新 | Forced-miss，Evict 更新 |
| Atomic | Forced-miss，Evict 更新 | Forced-miss，Evict 更新 |

Forced-miss 会按未命中处理，消除原有行的复用；Evict 使缓存行在服务完成后尽快可回收。没有预期复用的流式访问，可能受益于这种快速周转。

**给的信息：temporal 提示与访问 scope 一起决定 L1 命中和复用，进而决定多少流量进入 L2，以及缓存行占用多久。**

### 3.4.2 Input：输入 FIFO 能否继续接收请求

命令/地址/数据 FIFO 深度为 **10/6/4**。地址 FIFO 满时回压 TA。

**限制：短时间的请求突发可以缓冲，持续超出后续处理能力时会把停顿传回 CU。**

### 3.4.3 Tagram Mapping：地址能否分散到 4 个并行集合

同一 cycle 进来的线程地址会先合并成唯一缓存行；不同 cycle 的地址不做这种合并。随后把缓存行映射到 4 个 tagram，每个每 cycle 最多查一次。

```text
均匀分布时：最多 4 次 tagram lookup/cycle
全落同一 tagram：最多 1 次 lookup/cycle
单 tagram 可用容量：64 × 128 B = 8 KB
```

默认连续缓存行映射为 `0 → 2 → 1 → 3` 循环。周期性相隔 **512 B** 的地址会落到同一个 tagram，既使查找串行，又可能在其他集合仍空闲时就耗尽该集合的 64 条缓存行。

Buffer 指令的 cache swizzle 可以按 descriptor 中的 stride 改变映射，本文支持的相应 stride 为 256 B–8 KB 的 2 的幂；flat 指令不使用该机制。swizzle 改变热点的出现方式，例如 512 B swizzle 下，相隔 2 KB 的周期访问仍会落到同一集合。

**给的信息：32 KB 总容量和 4 lookup/cycle 总峰值能否被使用，取决于地址在四个集合间的分布。** TA 的线程地址合并与这里的同 cycle 缓存行合并是两个步骤。

### 3.4.4 地址转换：UTCL1 接口每 cycle 能接受多少转换

4 个 tagram 最多同时提出 4 个转换请求，TCP ↔ UTCL1 接口最多 **2 次探测/cycle**，3–4 个同时请求需要串行。

此外两个接口按 lock-step 工作，都能接收时才能继续发送；FIFO 满或一侧未准备好会阻塞 TCP。

**限制：即使没有严重 TLB miss，转换接口速率和排队也可能限制吞吐；miss 则进一步增加往返等待。**

### 3.4.5 Tagram Lookup：L1 查找和 L2 miss 发送速度不同

```text
L1 tagram 查找：最多 4 lookup/cycle
控制器接收 miss：最多 4 miss/cycle
向 L2 发送请求：最多 1 request/cycle
```

大量 miss 时，前面产生请求可能快于后面发出请求，需要缓冲与合并。返回数据准备好后，LFIFO 对应条目出队，缓存读取还需处理 RAM bank 冲突。

**给的信息：L1 本地查找峰值、L1→L2 请求发送峰值和数据返回能力分别约束不同阶段。**

### 3.4.6 Cache Controller：哪些资源耗尽会阻止前进

原文列出四类停顿来源：LFIFO 满、RFIFO 满、缓存 RAM 写入被返回数据路径阻塞，以及输入写数据 FIFO 为空。

**限制：控制器不能推进时，会继续回压 tagram、TA 和 CU；因此观察到 TA 停顿，也需要往 TCP 内部和下游找原因。**

### 3.4.7 Latency FIFO：能容纳多少等待数据返回的请求

LFIFO 深度为 **704 条目**，缓存 hit 和 miss 请求都会进入其中等待数据可用。表 8 按访问类型计算它能表示的缓存行数量：

| 访问类型 | 每 LFIFO 条目表示的数据量 | 每指令消耗条目 | 表 8 的缓存行数量 |
| --- | --- | ---: | ---: |
| `dword` Read | 4 DW = 16 B | 16 | 88 |
| `dwordx2` Read | 8 DW = 32 B | 16 | 176 |
| `dwordx4` Read | 16 DW = 64 B | 16 | 352 |
| `dword` Write | 2 CL | 1 | 1408 |
| `dwordx2` Write | 4 CL | 1 | 2816 |
| `dwordx4` Write | 8 CL | 1 | 5632 |

例如读操作按原文表格换算：

```text
dword Read：  704 × 16 B / 128 B = 88 CL
dwordx2 Read：704 × 32 B / 128 B = 176 CL
dwordx4 Read：704 × 64 B / 128 B = 352 CL
```

这是 **LFIFO 条目的请求数据表示能力**，不能直接当作 32 KB L1 的驻留容量，也不能当作最终独立 HBM 在途缓存行数；地址重叠、tagram 容量、合并和其他队列会继续施加约束。写入还采用不同的条目累积与合并方式。

**给的信息：相同 FIFO 深度下，访问宽度会改变能够表示的等待数据量。队列装满前可容纳多少并发请求，决定了能隐藏多长的延迟。** 这对应开头 `N = λT` 中的在途容量限制。

### 3.4.8 L2 Interface：L1 向 L2 的出口有多快

RFIFO 深度为 **64**，最多每 cycle 接收 4 个 TCP miss，请求发往 L2 的速率为 1 request/cycle。读请求大小为 128 B，写请求大小为 64 B。

```text
该请求接口的字节速率：
读：1 request/cycle × 128 B = 128 B/cycle
写：1 request/cycle ×  64 B =  64 B/cycle
```

L2 无法继续接收时会向 TCP 回压；可用 `TCP_TCR_TCP_STALL_CYCLES` 观察。

**3.4 合起来给的信息：缓存策略决定流向 L2 的请求量；tagram 和地址转换决定并行处理速度；LFIFO/RFIFO 决定能缓冲多少等待；L2 接口和回压决定请求能否继续前进。这些共同影响 3.2 的有效并发和 wave 寿命。**

## 3.5 Vector-L2 Cache：XCD 汇总吞吐、仲裁损失和 SoC 出口限制

L2 汇总 XCD 内 CU、指令缓存、标量 L1、UTCL2 等客户端的请求。它有 **16 个独立通道、合计 4 MB**，每通道为 128 sets × 16 ways，缓存行为 128 B。

### 3.5.1 Cache Policy：多少请求可以在 L2 命中

- **LRU：** 可命中，访问更新缓存行年龄，支持继续复用。
- **Streaming：** 可命中，但不刷新年龄，使缓存行更快可替换，适合较少复用的数据。
- **Bypass：** 强制外部获取、临时存放，服务后失效，不保留用于后续复用。

具体选择还取决于访问 scope 和分区模式：

| Read/Write 的 scope 与配置 | Temporal | Non-temporal |
| --- | --- | --- |
| Wave/Group | LRU | Streaming |
| Device，CPX | LRU | Streaming |
| Device，SPX/DPX | Bypass | Bypass |
| System | Bypass | Bypass |
| Atomic | Bypass | Bypass |

**给的信息：L2 策略决定请求是否在 XCD 内完成，还是继续经过 fabric/LLC/HBM。** 因此 non-temporal 在 L2 可以仍然命中，具体行为由 scope 和分区配置共同决定。

### 3.5.2 Arbitration Network：理论带宽乘仲裁效率，再折算每 CU

客户端通过读 128 B、写 64 B 的总线连接通道，每通道每 cycle 服务 1 个请求：

```text
L2 峰值读带宽 = 16 × 128 = 2048 B/cycle/XCD
L2 峰值写带宽 = 16 ×  64 = 1024 B/cycle/XCD

L2 有效带宽 = 峰值带宽 × 仲裁效率
每 CU 平均份额 = L2 有效带宽 / 每 XCD 活跃 CU 数
```

原文表 10 对 MI300X 给出读仲裁效率约 90%、写约 58%、38 CU/XCD：

```text
读：2048 × 90% ≈ 1843 B/cycle/XCD
    1843 / 38 ≈ 48.5 B/cycle/CU

写：1024 × 58% ≈ 594 B/cycle/XCD
    594 / 38 ≈ 15.6 B/cycle/CU
```

**算出的信息：经过仲裁后，实际有效吞吐低于总线宽度给出的峰值。** 每 CU 数值是按客户端流量较均匀的方式分摊所得；通道热点、其他 L2 客户端竞争还会改变实际份额。

表 10 中 MI350 的写仲裁效率为 75%，本文模型给出 `768 / 32 = 24 B/cycle/CU`。该列属于原文所列型号参数，不能直接代入当前 CDNA 4 MI355X。

### 3.5.3 Input Buffer：请求和写数据能否及时配合

写/原子请求与数据异步到达，通道仲裁决定哪个客户端发送数据。一个客户端同 cycle 只能赢得向一个通道写数据的机会；16 个通道可并行服务请求。

**限制：16 通道的汇总峰值依赖请求分布，以及写数据能否及时到达。**

### 3.5.4 LFIFO：L2 能缓存多少等待中的事务

L2 有共享的 **512 深 LFIFO**，每客户端另有 **7 深 FIFO**，用于容纳等待并保持客户端间并发。

**给的信息：L2 也需要足够的在途容量才能隐藏 SoC 往返延迟。** 原文没有在这里给出统一的“512 条目等于多少字节带宽”公式；换算需要事务大小、条目占用方式和往返延迟。

### 3.5.5 Fabric Interface：XCD 出口有多宽，下游如何回压

每 L2 通道通过读写共享的 64 B 总线连接 SoC：

```text
L2 ↔ SoC 汇总出口 = 16 × 64 = 1024 B/cycle/XCD
```

因此本章模型的 L2 内部峰值读能力为 2048 B/cycle/XCD，而继续流向 SoC 的出口为 1024 B/cycle/XCD。L2 命中读与 L2 miss 后访问外部内存，受约束的路径不同。

SoC 侧相应缓冲资源耗尽时，L2 被回压，并可能继续传到 L1 和 CU。

**3.5 给出的信息：L2 内部仲裁决定有效汇总吞吐，SoC 出口决定继续向外的通路上限，下游延迟和回压决定队列多久释放。** 明显 SoC stall 是外部路径受限的线索，仍应结合流量目标、带宽和延迟判断。

## 按节对照：第三章各部分到底得到什么

| 部分 | 计算或分析的对象 | 得到的关键结果 |
| --- | --- | --- |
| **3 开头** | 在途数量、吞吐、停留时间 | `N = λT`；高带宽需要足够多有效在途请求来覆盖延迟 |
| **3.1.1** | chunk 到来周期与 SPI 派发时间 | `C_p = #XCD × C`；`C_s = C/#SPI × WG × 4`；本文得到 `WG < #XCD` 的供给限制 |
| **3.1.2** | wave 寿命与 slot 轮转补充周期 | `T_p = #TotalCUs × W_o / WG`；`T_s < T_p` 时可能受 wave 派发限制 |
| **3.2** | 每 CU 的 wave 请求量与周转时间 | `BW_CU = min{W_o × n × R / (T_d + T_s), 64}` |
| **3.3** | TA 处理一条指令地址的时间 | 合并为 4 cycles、未合并为 16 cycles；不同指令对应 16/32/64 B/cycle 的每 CU 上限 |
| **3.4** | L1 各部件的速率、缓存复用、队列和热点 | 4 tagram lookup/cycle、2 UTCL1 probe/cycle、1 L2 request/cycle；FIFO 和地址分布限制有效在途容量 |
| **3.5** | L2 汇总吞吐、仲裁效率及外部出口 | 内部读/写峰值 2048/1024 B/cycle/XCD，乘仲裁效率得到有效值；SoC 出口为 1024 B/cycle/XCD |

读第三章时，始终可以沿着一条链检查：**前端有没有及时提供 wave → wave 能不能持续发有效请求 → TA/L1/L2 能不能接收和处理 → 下游能不能及时返回并释放资源。** 一个下游队列满，会逐级回压，最终表现为 CU 发射变慢、wave 寿命增长、实际带宽下降。
