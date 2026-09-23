# Original Performance Take-Home：913-cycle 结果

`perf_takehome.py` 是提交代码，本文记录来源、结构、结果和复现方法。代码依赖原挑战仓库的 `problem.py` 和官方测试，因此这两个文件本身不能独立运行。

## 目标与结果

挑战要求优化 `KernelBuilder.build_kernel`，在未修改的冻结模拟器和官方测试上尽量降低周期数。官方工作负载为 `(forest_height, n_nodes, batch_size, rounds) = (10, 2047, 256, 16)`。

| 保留节点 | 官方周期数 | 验证 |
| --- | ---: | --- |
| 本轮起点 | 982 | 官方测试 9/9 |
| 上一保留版本 | 959 | 官方测试 9/9 |
| 本仓库版本 | **913** | 官方测试 **9/9**，九次测量均为 913 cycles |

913 比 982 少 69 cycles（7.03%），比 959 少 46 cycles（4.80%）。本轮设定的严格目标是 **低于 900 cycles**；913 尚未达到该目标。实验中出现过低于 900 的资源下界，但没有得到通过完整冻结机与官方测试的低于 900 周期内核，因此不计作实际成绩。

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

最后一条命令应通过 9 个官方测试，周期输出均为 913。

在本次验证使用的 Windows + WSL 环境中，复制代码后从原挑战目录运行：

```powershell
./submission_tests.py
```
