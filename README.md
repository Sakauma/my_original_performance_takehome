# Original Performance Take-Home：913-cycle 结果

本仓库按[参考项目](https://github.com/BasicCoder/original_performance_takehome_work)的两文件形式整理：`perf_takehome.py` 是提交代码，本文记录来源、结构、结果和复现方法。代码依赖原挑战仓库的 `problem.py` 和官方测试，因此这两个文件本身不能独立运行。

## 目标与结果

挑战要求优化 `KernelBuilder.build_kernel`，在未修改的冻结模拟器和官方测试上尽量降低周期数。官方工作负载为 `(forest_height, n_nodes, batch_size, rounds) = (10, 2047, 256, 16)`。

| 保留节点 | 官方周期数 | 验证 |
| --- | ---: | --- |
| 本轮起点 | 982 | 官方测试 9/9 |
| 上一保留版本 | 959 | 官方测试 9/9 |
| 本仓库版本 | **913** | 官方测试 **9/9**，九次测量均为 913 cycles |

913 比 982 少 69 cycles（7.03%），比 959 少 46 cycles（4.80%）。本轮设定的严格目标是 **低于 900 cycles**；913 尚未达到该目标。实验中出现过低于 900 的资源下界，但没有得到通过完整冻结机与官方测试的低于 900 周期内核，因此不计作实际成绩。

## 实现与来源

官方形状使用公开仓库 [`littletiny/original_performance_takehome`](https://github.com/littletiny/original_performance_takehome) 的 [commit `816687ed901032d73b74101a38b9eabd04695002`](https://github.com/littletiny/original_performance_takehome/commit/816687ed901032d73b74101a38b9eabd04695002) 中的 913-cycle 程序。本仓库现在用结构化生成器构造同一条程序：`build()` 发出树遍历、哈希、加载与存储的具名算法图；`make_config()` 给出布局策略和少量具名标量化例外；按轮次、组和操作阶段组织的时序计划记录离线搜索已选出的运行周期；`allocate()` 将虚拟寄存器分配到 scratch，`lower()` 展开分支表并生成 10,537 个静态 bundle。该文件 6,295 行，其中具名配置与时序计划 3,654 行、算法和接口等代码 2,641 行；使用 Python 标准库，不依赖压缩数据、反序列化、NumPy 或外部生成脚本。

这一重构基于本地历史研究中的算法图与指令生成流程，辅以公开 checkpoint 对应的离线调度结果；它没有重建原作者的优化搜索器，也不声称是原作者的原始高层源码。**913-cycle 核心仍归属上述公开 checkpoint，本轮成果是可维护的等价生成方式，不是独立原创性能优化。**编辑算法图、布局或时序计划后，需要重新验证相互匹配和最终指令。官方形状走此生成器，其他形状仍走原有的 `build_kernel_baseline` 标量实现；`CFG["USE_EMBEDDED"]` 保留历史开关。首次构建会编译并缓存私有母本，此后每次调用都返回全新的 bundle 字典和 slot 列表。913 是模拟器执行周期，不包括 Python 构建时间。本仓库保留了源码顶部的 Anthropic 版权与使用声明原文；其中写明允许修改和使用，但不允许发布或再分发解答。

阅读源码时，可从 `build()` 的哈希与树路径开始，再看 `make_config()` 中的布局参数，随后查 `ROUND_STAGE_CYCLES` 和 `SETUP_AND_TAIL_CYCLES` 中的具名周期，最后看 `allocate()` 与 `lower()` 如何生成实体指令。`h1`、`h2`、`h4`、`h5`、`h6` 是哈希流水线阶段；`.laneN` 指八字向量的第 N 个标量 lane。

本仓库 `perf_takehome.py` 的 SHA-256 为 `91B5F8625566D529B2107B569001778E508F6B416D78FF3B061F5E45F3753674`。它只使用 Python 标准库和原挑战的 `problem.py`，运行时不依赖额外的搜索工具或外部文件。

## 复现

以下命令在普通 Linux shell 中执行。先取得本仓库和原挑战的固定测试版本，再将本仓库的唯一代码文件放入原挑战目录：

```bash
git clone https://github.com/Sakauma/my_original_performance_takehome.git
git clone https://github.com/anthropics/original_performance_takehome.git
cd original_performance_takehome
git checkout 5452f74bd977807ac2e74f3d29432b9df6f25197
cp ../my_original_performance_takehome/perf_takehome.py ./perf_takehome.py
git diff --exit-code -- problem.py tests/
python tests/submission_tests.py
```

最后一条命令应通过 9 个官方测试，周期输出均为 913。`git diff --exit-code -- problem.py tests/` 应无输出，确认测试和模拟器没有改动。测试版本为原挑战仓库 `main` 的 `5452f74bd977807ac2e74f3d29432b9df6f25197`；与本次验证使用的基准版本一致。

在本次验证使用的 Windows + WSL 环境中，复制代码后从原挑战目录运行：

```powershell
wsl -d ubuntu2004 -- /home/sakauma/data/miniconda3/envs/egor/bin/python tests/submission_tests.py
```

上一版压缩表示曾验证官方形状的多个随机种子与边界输入，以及若干非官方形状的兼容路径。`problem.py`、`tests/frozen_problem.py` 和 `tests/submission_tests.py` 均未修改。

结构化生成器另经逐项核对：官方形状生成的 10,537 个完整 bundle（含 pause 位置）与上一版全列表相同；`KernelBuilder` 入口和多次构建的实例隔离也相同。正式官方测试结果见上表。
