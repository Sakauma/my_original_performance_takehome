# Original Performance Take-Home：899-cycle 结果

`perf_takehome.py` 是提交代码，本文记录来源、结构、结果和复现方法。仓库延续只保留提交源码和本说明的形式；代码仅依赖 Python 标准库和原挑战仓库的 `problem.py`，验证使用原挑战的官方测试。

## 目标与结果

挑战要求优化 `KernelBuilder.build_kernel`，在未修改的冻结模拟器和官方测试上尽量降低周期数。官方工作负载为 `(forest_height, n_nodes, batch_size, rounds) = (10, 2047, 256, 16)`。

| 保留节点 | 官方周期数 | 验证 |
| --- | ---: | --- |
| 早期保留版本 | 982 | 官方测试 9/9 |
| 中间保留版本 | 959 | 官方测试 9/9 |
| 本仓库上一版本 | 913 | 官方测试 9/9 |
| 本仓库当前版本 | **899** | 官方测试 **9/9**，九次测量均为 899 cycles |

899 比 913 少 14 cycles（1.53%），相对官方 147734-cycle 基线的加速比为 **164.33×**，已达到 **低于 900 cycles** 的严格目标。这里的周期数是冻结模拟器实际执行结果；Python 端构建内核的耗时不计入周期成绩。

## 实现与适用范围

这个版本专用于官方计分工作负载：树高 10、2047 个节点、256 个输入、16 轮、初始索引全为 0，校验最终输出值。树节点值和输入值均在运行时读取，不依赖测试种子或预存答案。不承诺支持任意树高、批量尾部、轮数或非零初始索引。

主要优化包括：

- 对哈希计算做 32 位模运算等价变换，保留 XOR 编码的中间值以减少算术操作。
- 缓存浅层节点，将两层子树重排成连续数据包，减少依赖加载和索引计算。
- 将部分二选一操作改写为临时内存中的条件写入与向量加载，另一些使用等价的仿射计算，平衡 FLOW 与算术引擎负载。
- 通过 SSA 操作图、列表调度、标量/向量转换和按 lane 的寄存器复用生成指令；最后四组输出重排末级 XOR 的依赖关系。

源码直接生成普通 ISA 指令，不依赖外部搜索脚本、保存的程序或调度表。`KernelBuilder.build_kernel` 构建遍历和哈希计算，`Graph.schedule` 调度操作，`startup_synthesize` 生成启动常量，`post_transform` 改写选择操作，`reassociate_final` 缩短末级依赖链，`allocate` 分配物理 scratch。

内核使用 **1530 / 1536** 个 scratch word，保留原有单核、向量长度 8，以及每周期 ALU/VALU/LOAD/STORE/FLOW 为 12/6/2/2/1 的指令槽限制。最终程序包含 899 个指令束，各引擎操作数依次为 10655/5365/1756/832/862。

内核按官方只校验最终输出值的约定复用已消费的内存：浅层树重排到地址 16–255，末轮浅层镜像位于 0–15，四个临时选择缓冲区位于未使用的输入索引区 2054–2117。最终输出值仍位于 2310–2565；原始头部、浅层树和输入索引区不予保留。scratch 的 0–7 始终保留为架构初始化的零值。

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

最后一条命令应通过 9 个官方测试，九次周期输出均为 899。首次构建内核可能耗时数十秒。

在原生 Windows PowerShell 中，完成两个仓库的克隆后可执行：

```powershell
Set-Location original_performance_takehome
git checkout 5452f74bd977807ac2e74f3d29432b9df6f25197
Copy-Item ..\my_original_performance_takehome\perf_takehome.py .\perf_takehome.py
git diff --exit-code -- problem.py tests/
python tests/submission_tests.py
```

## 验证与来源

源码提取自 `performance_takehome_899.zip`，保持与压缩包内的提交文件逐字节一致。原始 LF 换行文件的 SHA256 为：

```text
ab5c199d255bab0d8ca2a31f124dd75ac0f064ac0fe343eae21a5a472a671156
```

2026-10-02 使用 Windows、Python 3.12.14 和上述固定官方提交重新验证：仅替换 `perf_takehome.py`，官方测试 9/9 通过，九次测量均为 899 cycles。`problem.py`、冻结模拟器及 `tests/` 均保持原样。压缩包内各文件的 SHA256 校验也全部通过。

本次另行核对了指令槽限制、scratch 地址边界、同周期 scratch 写冲突及保留零值区，并补测 16 组输入（32 位随机值、全相同边界值、逐位模式和交替模式）；所有输出均与冻结参考实现一致，周期数均为 899。
