# Original Performance Take-Home：891-cycle 结果

`perf_takehome.py` 是提交代码，本文记录来源、结构、结果和复现方法。仓库延续只保留提交源码和本说明的形式；代码仅依赖 Python 标准库和原挑战仓库的 `problem.py`，验证使用原挑战的官方测试。

## 目标与结果

挑战要求优化 `KernelBuilder.build_kernel`，在未修改的冻结模拟器和官方测试上尽量降低周期数。官方工作负载为 `(forest_height, n_nodes, batch_size, rounds) = (10, 2047, 256, 16)`。

| 保留节点 | 官方周期数 | 验证 |
| --- | ---: | --- |
| 早期保留版本 | 982 | 官方测试 9/9 |
| 中间保留版本 | 959 | 官方测试 9/9 |
| 前期上传版本 | 913 | 官方测试 9/9 |
| 本仓库上一版本 | 899 | 官方测试 9/9 |
| 本仓库当前版本 | **891** | 官方测试 **9/9**，九次测量均为 891 cycles |

891 比 899 少 8 cycles（0.89%），相对官方 147734-cycle 基线的加速比为 **165.81×**。这里的周期数是冻结模拟器实际执行结果；Python 端构建内核的耗时不计入周期成绩。

## 实现与适用范围

这个版本专用于官方计分工作负载：树高 10、2047 个节点、256 个输入、16 轮、初始索引全为 0，校验最终输出值。树节点值和输入值均在运行时读取，不依赖测试种子或预存答案。不承诺支持任意树高、批量尾部、轮数或非零初始索引。

主要优化包括：

- 对哈希计算做 32 位模运算等价变换，保留 XOR 编码的中间值以减少算术操作。
- 缓存深度 0–2 的节点，将深度 3–4 和 5–7 的子树重排成连续数据包，结合紧凑编码指针、部分深度 6 地址复用和末轮选择减少遍历操作。
- 将部分二选一、四选一操作改写为临时内存中的条件写入与向量加载，平衡 FLOW 与算术引擎负载。
- 将条件写入的丢弃目标合并到已无后续读取的内存地址 0，并复用已消费的头部内存完成部分广播，减少指针设置和算术操作。
- 通过 SSA 操作图、列表调度、标量/向量转换和按 lane 的寄存器复用生成指令；保留最后四组输出的末级 XOR 重排。

源码直接生成普通 ISA 指令，不依赖外部搜索脚本、保存的程序或调度表。`KernelBuilder.build_kernel` 调用 `compile_program`，由 `FrontendBuilder` 构建遍历和哈希计算；`Graph.schedule` 调度操作，`improve` 改写选择操作，`dead_zero_sink` 与 `header_broadcast` 处理内存复用，`shape_allocate` 分配物理 scratch。

内核使用 **1390 / 1536** 个 scratch word，比 899-cycle 版本减少 140 个，保留原有单核、向量长度 8，以及每周期 ALU/VALU/LOAD/STORE/FLOW 为 12/6/2/2/1 的指令槽限制。最终程序包含 891 个指令束，各引擎操作数依次为 10573/5285/1706/1327/805。

内核按官方只校验最终输出值的约定复用已消费的内存：树数据包位于地址 8–255，头部地址 0–7 暂用于广播，四个临时选择缓冲区位于输入索引区的 2054–2061、2070–2077、2086–2093、2102–2109。最终输出值仍位于 2310–2565；原始头部、部分树和输入索引区不予保留，也不输出最终索引。scratch 的 0–7 始终保留为架构初始化的零值。

内存地址 0 在广播结束后作为条件写入的丢弃目标，允许同周期重复写入这个不再读取的地址。地址分析确认其最后一次可能读取为第 64 周期、首次可能条件写入为第 94 周期（均从 0 计数）；原始模拟器允许重复目标写入，且此处结果不依赖写入顺序。审计仅允许这一例外，其余内存写冲突和所有 scratch 写冲突均检查并拒绝。

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

最后一条命令应通过 9 个官方测试，九次周期输出均为 891。首次构建内核可能耗时数分钟。

在原生 Windows PowerShell 中，完成两个仓库的克隆后可执行：

```powershell
Set-Location original_performance_takehome
git checkout 5452f74bd977807ac2e74f3d29432b9df6f25197
Copy-Item ..\my_original_performance_takehome\perf_takehome.py .\perf_takehome.py
git diff --exit-code -- problem.py tests/
python tests/submission_tests.py
```

## 验证与来源

源码提取自 `performance_takehome_891.zip`，保持与压缩包内的提交文件逐字节一致。原始 LF 换行文件的 SHA256 为：

```text
a19ec2e60a5a205d29f0f8da1d79ee725c4b3ed62703dbc85e09e64f101745ca
```

2026-10-02 使用 Windows、Python 3.12.14 和上述固定官方提交重新验证：仅替换 `perf_takehome.py`，官方测试 9/9 通过，九次测量均为 891 cycles。`problem.py`、冻结模拟器及 `tests/` 均保持原样。压缩包内各文件的 SHA256 校验也全部通过。

本次另行运行随包的独立审计脚本：277 组输入（32 位与 31 位随机值、边界值组合、边界循环、逐位模式和相同输入值批次）全部与冻结参考实现一致，周期数均为 891。审计同时核对指令槽限制、地址边界、scratch 值身份、调度依赖、保留零值区和上述丢弃地址的使用条件。
