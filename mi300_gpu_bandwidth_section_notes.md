# 《AMD Instinct MI300 GPU Memory Bandwidth》逐节重点

- 原文：[mi300_gpu_bandwidth.pdf](./mi300_gpu_bandwidth.pdf)
- 作者：Xiaomin Lu（AMD），2024 年 6 月，共 30 页。
- 页码：以下均指 PDF 页码。
- 阅读范围：文中的“MI300”泛指 MI300A、MI300X、MI308 和 MI350 系列；具体频率和 CU 数以相应型号为准。

## 文档索引

所有文件位于原 PDF 同一目录 `/root/workspace/trans_github/`。

| 文档 | 内容 |
| --- | --- |
| [MI355X 完整结构与各级带宽图](./mi355x_theoretical_bandwidth.html) | 单张完整结构图、各层带宽与 SoC 最小值公式 |
| [第三章详细总结](./mi300_gpu_bandwidth_chapter3_notes.md) | 在途请求、派发公式、每 CU 带宽、TA/L1/L2 的限制 |
| [第四章详细总结](./mi300_gpu_bandwidth_chapter4_notes.md) | 指标含义、计算方式、瓶颈判断和排查方向 |
| [第五章总结](./mi300_gpu_bandwidth_chapter5_notes.md) | 作者的结论，以及第二至第四章的关系 |

## 全文主线

内存请求经过 **CPC/SPI 派发 → CU/VMEM → TA → TCP（向量 L1）→ TCC（L2）→ SoC/LLC → HBM**。最终带宽同时取决于各层通路的硬件上限，以及程序能否持续提供足够多的在途请求。第 2 章确定各层上限，第 3 章说明请求怎样被派发、排队和回压，第 4 章给出用性能计数器定位瓶颈的方法。

文中主要按固定频率建模，并把 SoC 侧的带宽与延迟细节作为黑箱处理。以下数值是文中的标称值或模型上限，不能直接当成某个 kernel 的实测带宽。

## 1. Introduction：为什么理论 HBM 带宽难以用满（第 1–2 页）

- **重点：** MI300 系列同时提供大量 CU 和高 HBM 带宽。例如，MI300X 有 304 个 CU、约 5.12 TB/s HBM 带宽；MI308 用较少的 CU 提供同量级 HBM 带宽。整机 HBM 数字之外，还需要关心每个 CU 实际能获得多少带宽。
- **结论：** 程序既要受硬件带宽“屋顶”约束，也要发出足够多的请求来隐藏内存延迟。软件访存形式、硬件队列、缓存行为和动态频率都会影响最终数值。因此分析不能只看 HBM 规格。

## 2. MI300 Theoretical Bandwidth：各层理论上限（第 2–6 页）

### 2.1 Performance Architecture：请求经过哪些硬件（第 2–4 页）

- SoC 由 AID、XCD 等多个 die 组成。每个 XCD 有 4 个 Shader Engine（SE），共享一个 16 通道、4 MB 的 L2。每个 CU 有独立的 32 KB 向量 L1。
- 一个 CU 有 4 个 SIMD；一个 wave 有 64 个线程；每个 SIMD 最多驻留 8 个 wave，因此一个 CU 最多驻留 32 个 wave。**4 个 SIMD 共用一条 VMEM 通路**，所有外部访存都需要经过它。
- CU 可以并行使用 VALU、标量、LDS、分支和 VMEM 等管线。文中给出的 VMEM/VALU 发射节奏是每 4 cycle 一条指令；不同类别的计算与访存可在不同管线重叠。
- 前端 CPC 在 XCD/SE 之间分配工作组，SPI 在资源允许时展开和派发 wave。
- **本节结论：** “CU 数多”和“wave 驻留多”都不直接等于带宽高。还要看共享 VMEM、各级缓存和前端派发是否跟得上。

### 2.2 Theoretical Bandwidth：沿路径逐层取最小值（第 4–6 页）

- CU ↔ 向量 L1 的数据通路上限是 **64 B/cycle/CU**。
- 单个向量 L1 ↔ L2 的读、写接口分别是 **128 B/cycle** 和 **64 B/cycle**。
- 每个 XCD 的 L2 有 16 个通道：内部理论读、写峰值分别为 **2048 B/cycle** 和 **1024 B/cycle**；L2 ↔ SoC fabric 的汇总通路是 **1024 B/cycle/XCD**。
- 文中表 3 的标称汇总数值如下。CU/XCD 汇总值是沿各自接口计算的理想上限，不能与实际 HBM 流量画等号。

| 型号 | CU 数 | 标称 SCLK | CU 通路汇总 | XCD→SoC 汇总 | HBM 标称上限 |
| --- | ---: | ---: | ---: | ---: | ---: |
| MI300X | 304 | 1.30 GHz | 25.293 TB/s | 10.650 TB/s | 5.120 TB/s |
| MI308 | 80 | 1.42 GHz | 7.270 TB/s | 5.816 TB/s | 5.120 TB/s |
| MI350 | 256 | 1.00 GHz | 16.384 TB/s | 8.192 TB/s | 8.192 TB/s |

- **本节结论：** 整机理论带宽受经过的层级共同约束，文中概括为 `SoC_BW = min(CU_BW, XCD_BW, LLC_BW, HBM_BW)`。分析时要把单 CU、单 XCD、整机，以及读/写接口分开。

## 3. MI300 Attainable Bandwidth：实际请求能否填满通路（第 6–19 页）

### 第 3 章总论：排队与延迟隐藏（第 6–7 页）

- 文中把 GPU 视为由派发器、CU、各级缓存、fabric 和 HBM 串成的排队系统，用 Little 定律 `N = λT` 理解在途请求数 `N`、吞吐量 `λ`、停留时间 `T` 的关系。
- **结论：** 提高带宽通常需要维持足够多的**在途内存请求**，并减少请求和 wave 的等待时间。驻留 wave 数只是其中一个条件：wave 没有继续发请求，或被下游回压时，高 occupancy 仍可能对应低带宽。

### 3.1 Frontend Workgroup and Wave Dispatch：前端能否持续供给 wave（第 7–8 页）

#### 3.1.1 Workgroup Dispatch：CPC 的工作组派发

- CPC 把 kernel 的工作组按 chunk 分给 XCD，并在一个 XCD 的 4 个 SPI 之间轮转。按文中的均匀派发模型，若每工作组 wave 数 `WG < XCD 数`，SPI 处理完当前分配后可能等待下一批工作组，形成前端限制。
- MI300X 有 8 个 XCD，故文中的界限是 **少于 8 wave/工作组**；对 4 个 XCD 的 MI308，按相同公式界限是少于 4 wave/工作组。
- **本小节结论：** 小工作组可能使多 XCD 系统的前端供给不足。这个条件来自文中的派发模型，应结合实际工作量和计数器验证。

#### 3.1.2 Wave Dispatch：SPI 的资源与补位速度

- SPI 只有在一个工作组所有 wave 的 VGPR、SGPR、LDS、barrier、wave slot 等需求都能满足后，才会派发该工作组。
- 文中给出 wave slot 的补位周期 `T_p = 总 CU 数 × 目标驻留 wave 数 / 每工作组 wave 数`。若 wave 寿命 `T_s < T_p`，wave 结束得比 SPI 补位快，形成 wave 派发限制。
- 示例：MI300X 的 304 个 CU、4 wave/工作组、目标 32 wave/CU 时，`T_p = 304 × 32 / 4 = 2432 cycle`。另一个示例是 4 wave/工作组且每组用 32 KB LDS；64 KB LDS/CU 只容纳 2 组，因此最多 8 wave/CU。
- **本小节结论：** 需要同时检查静态资源决定的驻留上限，以及 SPI 能否及时补充结束的 wave。

### 3.2 Compute Unit：每 CU 带宽模型（第 8–10 页）

- 一个 CU 的 4 个 SIMD 共享 VMEM。命令和地址 FIFO 各深 48，写数据 FIFO 深 64；64 线程 wave 按 4 个阶段传给向量 L1。没有下游回压时，一条 VMEM 指令约每 4 cycle 传完；有回压时，指令发射间隔和 wave 寿命都会增加。
- 文中用以下稳态模型描述每 CU 带宽：

  `BW_CU = min[W_o × n × R / (T_d + T_s), 64] B/cycle`

  其中 `W_o` 是平均驻留 wave 数，`n` 是每 wave 的访存指令数，`R` 是每条指令请求的字节数，`T_d` 是派发间隔，`T_s` 是 wave 执行寿命。例如 64 个线程各请求 8 B 的 `buffer_load_dwordx2`，`R = 512 B`。

- **本节结论：** 提高有效驻留 wave、增加有用的访存请求或每指令请求量、缩短派发及访存等待，都有助于提高带宽；最高仍受 64 B/cycle/CU 的数据通路约束。增加驻留 wave 也会增加对共享 VMEM 的竞争，需看最终吞吐而非单独追求 occupancy。

### 3.3 Texture Addressing Unit（TA）：地址合并与指令粒度（第 10–12 页）

- TA 从 CU 接受 VMEM 指令，最多每周期向 TCP 传 4 个地址。对满足条件的相邻 DWORD 地址，TA 可以把最多 4 个线程的访问编码为一个地址：一个 wave 最快约 **4 cycle** 传完。不能这样合并的 64 个线程地址需要 **16 cycle**。
- 文中讨论的 `buffer_load_dwordx2/x4` 不支持上述 TA 地址合并。连续的未合并指令可能填满 TA 缓冲，随后回压 CU；文中举了 12 条指令的突发例子。

| VMEM 指令 | 每 wave 指令请求量 | TA 限制下的每 CU 吞吐上限 |
| --- | ---: | ---: |
| `buffer_load_dword`，未合并 | 256 B | 16 B/cycle |
| `buffer_load_dword`，已合并 | 256 B | 64 B/cycle |
| `buffer_load_dwordx2`，未合并 | 512 B | 32 B/cycle |
| `buffer_load_dwordx4`，未合并 | 1024 B | 64 B/cycle |

- **本节结论：** 如果访问模式和实际所需数据允许，用 `dwordx4` 代替多次较窄加载可提高这一环节的字节吞吐，并减少完成同样数据量所需的指令及部分下游查找。表中数值是 TA/CU 环节上限，不保证整机达到相应 HBM 带宽。这里的 TA 地址合并与 TCP 随后的缓存行合并是两个不同过程。

### 3.4 TCP：向量 L1 的缓存、转换和排队（第 12–17 页）

#### 3.4.1 vL1D Cache Policy：缓存策略

- L1 的 temporal 读在 wave/group scope 下可正常命中并更新 LRU；device/system scope 的读会按强制 miss、较快失效的行为处理。non-temporal 读也会强制 miss 并在服务后使缓存行可回收。
- 写入不复用现有缓存行：temporal 写执行强制 miss 后采用 LRU 更新，non-temporal 写采用较快失效；原子操作也按强制 miss/失效处理。
- **结论：** 判断 L1 是否会复用数据，要同时看指令的 temporal 属性和 memory scope。对几乎没有复用的流式数据，较快回收缓存行可能提高吞吐。

#### 3.4.2 Input：输入 FIFO

- TCP 的命令、地址和数据 FIFO 深度分别为 **10、6、4**；地址 FIFO 满会回压 TA，可从 `TCP_TCP_TA_ADDR_STALL_CYCLES` 观察。
- **结论：** TCP 输入队列是有限的；下游长期处理不过来最终会阻塞 CU 发指令。

#### 3.4.3 Tagram Mapping：四个 tagram 的地址分布

- TCP 有 4 个 tagram，每个每周期最多查一次；同一周期抵达的地址可按缓存行合并，但不同周期的地址不做这种合并。多个不同缓存行落入同一 tagram 时会串行。
- 默认映射使连续缓存行按 `0、2、1、3` 分布；周期性相隔 **512 B** 的地址可能反复落到同一个 tagram。对符合条件的 buffer 指令，可通过 cache swizzle 改变部分 2 的幂 stride 的映射；flat 指令不适用该开关。
- 单个 tagram 的 64 条缓存行全部被占用时会发生容量停顿。可结合 `TCP_ALLOC_STALL_CYCLES` 和 `TCP_TAGRAM0..3_REQ` 判断。
- **结论：** 总缓存容量和总查找峰值都不能掩盖单个 tagram 的热点；访存地址的 stride 和 swizzle 设置会影响实际查找吞吐。

#### 3.4.4 Virtual-to-Physical Address Translation：UTCL1 地址转换

- TCP 到 UTCL1 最多每周期进行 **2 次**转换探测；同时来自 3–4 个 tagram 的请求要串行。UTCL1 在途请求已满、miss 或下一级页表访问过慢时，也会向 TCP 回压。
- **结论：** TLB 命中率很高仍可能出现性能问题，还要看突发请求速率、miss 延迟和转换相关停顿。

#### 3.4.5 Tagram Lookup：查找与 L2 请求速率

- 四个 tagram 合计最多 4 次查找/cycle；miss 进入控制器和延迟隐藏队列，但向 L2 发请求的速度至多是 **1 次/cycle**。
- **结论：** L1 查找峰值与 L1→L2 miss 发送峰值是两种不同上限；大量 miss 会形成排队。

#### 3.4.6 Cache Controller：控制器停顿

- LFIFO 满、请求 FIFO 满、缓存 RAM 写入与回读冲突、输入写数据不足等情况都可阻塞控制器，并继续回压 tagram。
- **结论：** 看到 tagram 停顿后，还需要区分是本地资源不够，还是下游已经阻塞。

#### 3.4.7 Latency FIFO：容纳在途请求

- TCP 的 LFIFO 深 **704** 个条目。读请求按指令粒度占条目；表 8 计算出的读 `dword/x2/x4` 可容纳缓存行数约为 **88/176/352**。写入的条目组织方式不同。
- **结论：** FIFO 容量影响可隐藏的往返延迟，但实际并发还受 tagram 分布、请求类型和 L2 处理速度约束。

#### 3.4.8 L2 Interface：L1→L2 接口

- 64 深的 RFIFO 缓冲 L1 miss；到 L2 的发送速率是 1 请求/cycle。L2 对 TCP 的回压可通过 `TCP_TCR_TCP_STALL_CYCLES` 观察。
- **结论：** 若该回压明显，CU 的访存吞吐可能受 L2 或更下游限制。

### 3.5 Vector-L2 Cache：L2 的策略、仲裁和 SoC 接口（第 17–19 页）

#### 3.5.1 L2 Cache Policy：LRU、Streaming、Bypass

- LRU 允许命中并更新缓存行年龄；Streaming 仍允许命中，但不更新年龄，便于较快替换；Bypass 不保留数据供后续复用。
- wave/group scope 的 temporal 读写采用 LRU，non-temporal 采用 Streaming。device scope 在 CPX 分区下可采用相应策略，在 SPX/DPX 下改为 Bypass；system scope 与原子操作使用 Bypass。
- **结论：** non-temporal 并不总对应同一种 L2 行为；memory scope 与分区模式共同决定最终策略。

#### 3.5.2 Arbitration Network：16 通道仲裁

- 每个 XCD 有 16 个 L2 通道，单通道每周期处理一个请求。理想内部带宽为读 2048、写 1024 B/cycle/XCD；表 10 给出的有效仲裁效率约为读 **90%**、MI300X 写 **58%**、MI350 写 **75%**。
- MI300X 折算到 38 CU/XCD，有效 L2 前端读、写能力分别约 **48.5**、**15.6 B/cycle/CU**。通道间流量不均还会产生热点。
- **结论：** 尤其对写流量，L2 仲裁效率可能先于更下游的峰值成为限制；分析应看每通道流量。

#### 3.5.3 Input Buffer：请求与写数据

- 写/原子请求和数据可以异步到达，由输入缓冲协调；16 个通道可以并行处理各自的请求。
- **结论：** 汇总峰值依赖请求能否均匀抵达通道，以及写数据能否及时配合请求。

#### 3.5.4 LFIFO：隐藏 L2 后续延迟

- L2 有共享的 **512 深 LFIFO**，各客户端另有 **7 深 FIFO**，以保持多客户端并发。
- **结论：** L2 同样依赖足够的在途请求和队列空间来隐藏往返延迟。

#### 3.5.5 Fabric Interface：L2→SoC 回压

- 每个 L2 通道通过共享读写的 64 B/cycle 接口连到 SoC；SoC 缓冲或 credit 不足时会回压 L2。
- **结论：** 明显的 SoC 侧停顿可能表明外部内存路径接近限制，但还要结合实际流量目标和 L2 内部停顿判断。

## 4. MI300 Bandwidth Attainment Analysis：如何用计数器定位（第 19–25 页）

### 4.1 Perfmons and Performance Metrics：计数器看什么（第 19–20 页）

- **时钟类：** 测量 elapsed/busy cycle，作为利用率和带宽归一化的分母。
- **工作负载类：** 测指令混合、读写和不同请求大小，了解程序究竟发出了什么。
- **模块性能类：** 测各接口流量、缓存命中和回压；模块内部类再细分 FIFO、tagram、地址转换等原因。
- **本节结论：** 先用流量和忙碌时间定位大致层级，再用 stall 和模块内部计数器寻找原因。计数器的原始单位与实例聚合方式必须核对。

### 4.2 Performance Analysis：逐层诊断流程（第 20–25 页）

- 工作负载可能受计算吞吐、内存带宽、内存延迟或派发速度限制；优化后主瓶颈也会转移。图 9 的流程是诊断起点，不是能覆盖所有 kernel 的固定判定规则。

#### 4.2.1 Roofline Characterization：先测工作量与带宽

- 测 FLOPS、读写字节和运行时间。`TCC_EA_RDREQ_BW`、`TCC_EA_WRREQ_BW` 在表 13 中给的是**字节数**，需要除以运行时间才得到 GB/s；读请求可能是 32/64/128 B，写请求可能是 32/64 B。
- L2→SoC 流量可能包括 HBM、系统内存和 XGMI，不能未经区分就把它当作物理 HBM 流量。文中也给出该接口的读写延迟指标。
- **结论：** 先确认“程序真正经过哪条数据路径、跑到了什么水平”，再讨论瓶颈。

#### 4.2.2 GL2 Performance Analysis：检查 L2 与 SoC

- 看 L1→L2 请求方向、缓存策略、内存类型和 `TCP_TCR_TCP_STALL_RATE`。文中以超过约 **10%** 的回压作为值得怀疑 L2 的示例阈值。
- 看 16 个通道的逐通道请求，判断是否热点；看 `TCC_HIT_RATE`、`TCC_UTIL`、`TCC_TAG_STALL_RATE`，区分缓存复用、忙碌程度与内部/SoC 回压。
- 若外部带宽已很高（文中举例 **超过约 80%**）并伴随明显 tag stall，较支持外部带宽限制的判断。流式负载天然可能具有低 L2 hit rate。
- **结论：** 高 L2→L1 回压指向 L2 或更下游；进一步用通道平衡、命中率及 tag stall 来源区分原因。10%/80% 是文中举例，不是通用硬阈值。

#### 4.2.3 GL1 Performance Analysis：检查 TA、TCP 和 UTCL1

- **CU/SQ→GL1：** 看 flat/buffer、读/写/原子的指令构成，以及命令、地址、数据接口的回压。
- **TA：** `TA_BUF_ADDR_COALESCE_RATE` 理想约 100%，最差约 25%，对应一个 wave 的地址约 4 或 16 cycle 传完。结合汇编检查是否能用更合适的访问宽度。
- **TCP：** `TCP_TA_ADDR_STALL_RATE` 汇总了 TCP 及其下游造成的地址回压；再看 `TCP_SET_FULL_STALL_RATE`、`TCP_LFIFO_STALL_RATE` 和 `TCP_TAGRAM0..3_REQ`。若四个 tagram 命中率相近，最热/最冷请求超过约 **5 倍**可提示热点。
- **UTCL1：** 即使转换命中率接近 97%，突发请求过多、UTCL2 查询（文中称命中时往返约 82 cycle）或几百 cycle 的 page walk，仍可能放大等待。看 inflight 与 miss 相关停顿指标。
- **传播链：** TCP 排队或容量满 → TA 被回压 → `SQ_VMEM_CMD_STALL_RATE` 上升 → wave 寿命与 kernel 时间增加。若没有明显 L2→L1 回压，排查重点转向 L1 与其上游。
- **命中率的解释：** 顺序 `dwordx4` 读取时，两个相邻周期可分别访问同一 128 B 缓存行的两半，因此出现约 50% L1 hit；这本身不说明存在跨 wave 的数据复用。
- **结论：** 一个高的 `TCP_TA_ADDR_STALL_RATE` 只能说明 TCP 路径挡住了 TA，还需逐项拆解 tagram、FIFO、地址转换和 L2 回压。

#### 4.2.4 Compute Performance Analysis：检查 wave 时间花在哪里

- 将 wave 寿命拆成依赖等待、等待指令发射、指令发射/执行。大量依赖等待提示访存、barrier 等延迟没有被隐藏；大量发射等待提示目标管线争用。
- 使用 `SQ_WAVE_LIFE` 及 `SQ_WAIT_INST_VALU/LDS/VMEM/SCA` 等指标，判断主要等待发生在计算、LDS、VMEM 还是标量路径；再与 CU→TA 的回压对应。
- **结论：** “wave 活得很久”要继续分辨是等数据、等依赖，还是等管线发射，才能找到可改动的原因。

#### 4.2.5 Wave Dispatch Performance Analysis：检查驻留资源

- SPI 有 VGPR、SGPR、LDS、wave slot 等资源不足计数器；结合 ROCProf 的每 wave/工作组资源使用量解释 occupancy。各类资源计数器单位不同，原始数值不宜直接横比。
- 图 10 示例：每 wave 使用 112 个 archVGPR 和 128 个 accumVGPR，共 **240 VGPR**；每组约 **27 KB LDS**、4 wave/组。VGPR 限制到每 SIMD 2 wave，LDS 限制到每 CU 2 组，最终最多 **8 wave/CU**。
- **结论：** 找到真正卡住驻留 wave 的资源，才有依据调整寄存器、LDS 或工作组配置。

## 5. Summary：作者的总括（第 25 页）

- 第五章是一段全文总结，没有新增公式。作者概括了各层理论与可达到带宽、软件可见的性能模型，以及结合公开计数器的分析方法。
- 第二章给出经过路径的硬件上限；第三章解释请求并发、派发、执行、缓存与队列怎样影响实际吞吐；第四章通过测量确定哪些限制实际发生。
- **总论：** 逐层测流量、延迟与回压，确认瓶颈落在前端、CU、TA/L1、L2 还是 SoC/HBM，再选择相应的软件调整，并重新测量优化效果。
- 详细说明：[第五章总结](./mi300_gpu_bandwidth_chapter5_notes.md)。

## 附录：使用计数器时要记住什么（第 26–30 页）

### A. Performance Events（第 26–27 页）

表 12 列出本文用到的原始 perfmon 事件，覆盖 GPU 时钟、VMEM 发射与等待、TA/TCP 阻塞、UTCL1、L2 请求和 SPI 资源不足。**用途：** 查指标异常时回到原始计数器，确认事件到底测了请求数、cycle，还是资源实例数。

### B. Performance Metrics（第 26、28 页）

表 13 给出由原始计数器计算的带宽、命中率、stall rate 与延迟等指标。**用途：** 计算前核对指标的分子、分母和单位；名称带 `_BW` 的部分指标先表示字节量，还需要按时间归一化。

### C.1 Level Counters — Design（第 26、28–29 页）

level counter `L(t)` 表示某一时刻的在途事件数量；与之配对的 accumulator 记录 `I(T) = ∫₀ᵀ L(t)dt`。因此：

- 平均事件寿命或延迟：`I(T) / 事件总数`。
- 平均在途数量或 occupancy：`I(T) / 测量时长`。

**结论：** 计数事件总数只能回答“发生了多少次”；积分计数器还能回答“平均持续多久、平均同时有多少个”。

### C.2 Level Counters — Usage（第 29 页）

采集 level counter 时必须同时指定对应 accumulator，例如 `SQ_IFETCH`、`SQ_IFETCH_LEVEL` 和 `SQ_ACCUM_HIRES`；平均取指延迟由 accumulator 除以取指事件总数。表 14 列出取指、VMEM、SMEM、LDS 和 wave life 的配对关系。

### D. Multi-instance Counters（第 29–30 页）

L1/L2 有多个实例或通道。带 `_sum` 的派生计数器表示跨实例汇总；诊断 L2 通道热点时则需要逐实例数据。**结论：** 比较数值前先确定计数器的实例粒度。

## 数值与原文笔误提醒

1. **每 CU 峰值与整机 HBM 峰值分开看。** 按表 3 的 MI300X 标称值，5.12 TB/s 均摊到 304 个 1.3 GHz 的 CU，约为 `12.96 B/cycle/CU`。因此，3.3 中 `dwordx2` 的 `32 B/cycle/CU` 上限虽然只有 CU 数据通路上限的一半，不能单凭它断言整机 HBM 也只能达到一半；还要看活跃 CU 数与其他瓶颈。
2. **第 4 页表 2 的 MI300A CU 总数不自洽。** 表内同时列出 6 XCD、38 CU/XCD 和总计 288 CU；前两项相乘是 228。
3. **第 23 页的 TCP 容量“256 KB”与全文其他位置冲突。** 表 2、2.2 和 3.4 均给出 32 KB；4 个 tagram × 64 行 × 128 B/行也等于 32 KB。
4. **第 29 页的平均寿命/occupancy 公式分子应是积分值。** 原文先定义 `I(T) = ∫L(t)dt`，随后却印成 `L(T)/N` 与 `L(T)/T`；按定义和下一段 accumulator 示例，应分别使用 `I(T)/N` 与 `I(T)/T`。
