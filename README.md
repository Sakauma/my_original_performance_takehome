# Original Performance Take-Home：881-cycle 结果

`perf_takehome.py` 是提交代码，本文记录来源、结构、结果和复现方法。仓库延续只保留提交源码和本说明的形式；代码仅依赖 Python 标准库和原挑战仓库的 `problem.py`，验证使用原挑战的官方测试。

## 目标与结果

挑战要求优化 `KernelBuilder.build_kernel`，在未修改的冻结模拟器和官方测试上尽量降低周期数。官方工作负载为 `(forest_height, n_nodes, batch_size, rounds) = (10, 2047, 256, 16)`。

| 保留节点 | 官方周期数 | 验证 |
| --- | ---: | --- |
| 早期保留版本 | 982 | 官方测试 9/9 |
| 中间保留版本 | 959 | 官方测试 9/9 |
| 前期上传版本 | 913 | 官方测试 9/9 |
| 后续上传版本 | 899 | 官方测试 9/9 |
| 本仓库上一版本 | 891 | 官方测试 9/9 |
| 本仓库当前版本 | **881** | 官方测试 **9/9**，九次测量均为 881 cycles |

881 比 891 少 10 cycles（1.12%），相对官方 147734-cycle 基线的加速比为 **167.69×**。这里的周期数是冻结模拟器实际执行结果；Python 端构建内核的耗时不计入周期成绩。

## 实现与适用范围

这个版本专用于官方计分工作负载：树高 10、2047 个节点、256 个输入、16 轮、初始索引全为 0，校验最终输出值。树节点值和输入值均在运行时读取，不依赖测试种子或预存答案。不承诺支持任意树高、批量尾部、轮数或非零初始索引。

主要优化包括：

- 对哈希计算做 32 位模运算等价变换，保留 XOR 编码的中间值以减少算术操作。
- 缓存深度 0–2 的节点，将深度 3–4 和 5–7 的子树重排成连续数据包，结合编码指针、地址复用和兄弟节点预取减少遍历操作。
- 将部分二选一、四选一操作改写为临时内存中的条件写入与向量加载，平衡 FLOW 与算术引擎负载。
- 提前计算末轮分支条件，并重排选择前后的独立 XOR 运算，缩短依赖链；通过交换选择分支保持等价，最终输出值的计算不变。
- 使用布尔指针查找、从运行时头部构造常量、提前读取输入和十组重叠广播；保留已消费头部内存的复用及内存地址 0 的丢弃用途。
- 通过 SSA 操作图、标量/向量引擎选择、调度和按 lane 的寄存器复用生成指令，保留末级 XOR 重排。

源码直接生成普通 ISA 指令。`KernelBuilder.build_kernel` 调用 `compile_program`，从 `FrontendBuilder` 构建的遍历和哈希图出发，经 `build_candidate_graph` 应用变换，再由 `_candidate_schedule` 调度、`allocate` 分配物理 scratch。

本版本内嵌与输入数据无关的调度轨迹、DAG 发射时间和标量/向量模式提示。提示经过图指纹校验，编译器仍从源码重建操作，并检查依赖、指令槽容量和物理分配；提示不包含指令表、运行时树或输入值、预存答案。复现不需要外部搜索脚本、预编译程序或第三方求解器。

内核使用 **1444 / 1536** 个 scratch word，比 891-cycle 版本增加 54 个，保留原有单核、向量长度 8，以及每周期 ALU/VALU/LOAD/STORE/FLOW 为 12/6/2/2/1 的指令槽限制。最终程序包含 881 个指令束，各引擎操作数依次为 10469/5268/1690/1583/850。

内核按官方只校验最终输出值的约定复用已消费的内存：树数据包位于地址 8–255，头部地址 0–7 暂用于广播，四个临时选择缓冲区位于输入索引区的 2054–2061、2070–2077、2086–2093、2102–2109。最终输出值仍位于 2310–2565；原始头部、部分树和输入索引区不予保留，也不输出最终索引。scratch 的 0–7 始终保留为架构初始化的零值。

内存地址 0 在广播结束后作为条件写入的丢弃目标，允许同周期重复写入这个不再读取的地址。地址分析确认其最后一次可能读取为第 47 周期、首次可能条件写入为第 84 周期（均从 0 计数）；原始模拟器允许重复目标写入，且此处结果不依赖写入顺序。审计仅允许这一内存写冲突例外，其余内存写冲突和所有 scratch 写冲突均检查并拒绝。

十组重叠广播各保留完整的 12-word scratch 区间，利用原始模拟器同周期先读后写的语义复制相同标量。每组四次填充 lane 读取只流向无后续使用的输出；审计仅对这 40 次指定读取放宽已定义值检查，并验证别名关系、区间边界和后续使用。其余所有有效读取和 scratch 写入均严格检查。

## 复现

以下命令在 Linux、macOS 或 WSL 的 shell 中执行。先取得本仓库和原挑战的固定测试版本，再将本仓库的唯一代码文件放入原挑战目录：

```bash
git clone https://github.com/Sakauma/my_original_performance_takehome.git
git clone https://github.com/anthropics/original_performance_takehome.git
cd original_performance_takehome
git checkout 5452f74bd977807ac2e74f3d29432b9df6f25197
cp ../my_original_performance_takehome/perf_takehome.py ./perf_takehome.py
git diff --exit-code -- problem.py tests/
python tests/submission_tests.py
```

最后一条命令应通过 9 个官方测试，九次周期输出均为 881。已验证环境为 Python 3.12.14；本次完整官方测试耗时约 404 秒，主要用于首次构建内核，实际时间取决于机器。相同进程内对同一工作负载的构建会复用缓存。

在原生 Windows PowerShell 中，完成两个仓库的克隆后可执行：

```powershell
Set-Location original_performance_takehome
git checkout 5452f74bd977807ac2e74f3d29432b9df6f25197
Copy-Item ..\my_original_performance_takehome\perf_takehome.py .\perf_takehome.py
git diff --exit-code -- problem.py tests/
python tests/submission_tests.py
```

## 验证与来源

源码提取自 `performance_takehome_881.zip`，保持与压缩包内的提交文件逐字节一致。原始 LF 换行文件的 SHA256 为：

```text
0a2efeb06fbabba43357114d74c8968f714416b5caf06d496ee00a4aa95a359f
```

2026-10-04 使用 Windows、Python 3.12.14 和上述固定官方提交重新验证：仅替换 `perf_takehome.py`，官方测试 9/9 通过，九次测量均为 881 cycles。`problem.py`、冻结模拟器及 `tests/` 均保持原样。压缩包内各文件的 SHA256 校验也全部通过。

本次另行运行随包的独立审计脚本：277 组输入（32 位与 31 位随机值、边界值组合、边界循环、逐位模式和相同输入值批次）全部与冻结参考实现一致，周期数均为 881。审计同时核对指令槽限制、地址边界、有效 scratch 值身份、调度依赖、保留零值区、重叠广播、预取、内存顺序约束、算术变换和内嵌调度提示。这里的组件证明和测试结果不等同于整个内核的完整形式化证明。

生成程序的 `repr(instructions)` SHA256 为：

```text
9160f691bbf3d87edadefdb5436817bb9dff6db9638c0cdd81fcd6421cdc222c
```
