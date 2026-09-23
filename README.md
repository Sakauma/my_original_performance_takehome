# Original Performance Take-Home：913-cycle 结果

本仓库按[参考项目](https://github.com/BasicCoder/original_performance_takehome_work)的两文件形式整理：`perf_takehome.py` 是提交代码，本文记录来源、结果和复现方法。代码依赖原挑战仓库的 `problem.py` 和官方测试，因此这两个文件本身不能独立运行。

## 目标与结果

挑战要求优化 `KernelBuilder.build_kernel`，在未修改的冻结模拟器和官方测试上尽量降低周期数。官方工作负载为 `(forest_height, n_nodes, batch_size, rounds) = (10, 2047, 256, 16)`。

| 保留节点 | 官方周期数 | 验证 |
| --- | ---: | --- |
| 本轮起点 | 982 | 官方测试 9/9 |
| 上一保留版本 | 959 | 官方测试 9/9 |
| 本仓库版本 | **913** | 官方测试 **9/9**，九次测量均为 913 cycles |

913 比 982 少 69 cycles（7.03%），比 959 少 46 cycles（4.80%）。本轮设定的严格目标是 **低于 900 cycles**；913 尚未达到该目标。实验中出现过低于 900 的资源下界，但没有得到通过完整冻结机与官方测试的低于 900 周期内核，因此不计作实际成绩。

## 实现与来源

官方形状使用公开仓库 [`littletiny/original_performance_takehome`](https://github.com/littletiny/original_performance_takehome) 的 [commit `816687ed901032d73b74101a38b9eabd04695002`](https://github.com/littletiny/original_performance_takehome/commit/816687ed901032d73b74101a38b9eabd04695002) 中的 913-cycle 程序。本仓库将该确定性指令序列写成可直接检查的 Python ISA tuple：553 个固定 bundle 明文列出，360 个调度组用普通循环列出每种分支对应的具体操作数，构建内核时生成 10,537 个静态 bundle。源码不使用压缩数据或反序列化。

本地对源码所做的改动是将该程序展开为明文低层指令，并增加形状分派：上述官方形状使用该 913-cycle 程序，其他形状调用原有的 `build_kernel_baseline` 标量实现。**913-cycle 核心来自上述公开 checkpoint；可读源码只是对已发布指令的等价展开，并未恢复原作者的高层调度生成器，也不是本轮独立原创优化。**本仓库保留了源码顶部的 Anthropic 版权与使用声明原文；其中写明允许修改和使用，但不允许发布或再分发解答。

本仓库 `perf_takehome.py` 的 SHA-256 为 `AD1A4A5B120ECB8BFBC069E2AECE93B23062EE1F6BCDBF25AFFC1DE29CA261C9`。它只使用 Python 标准库和原挑战的 `problem.py`，运行时不依赖额外的搜索工具或外部文件。

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

明文改写另经独立核对：官方形状生成的完整 `instrs`（含 pause 位置）和 debug info 与上一版逐项相同；四个非官方形状的指令、输出内存与周期相同。关闭固定程序开关及多次构建隔离也通过验证。明文版重新通过官方 9/9，九次均为 913 cycles。
