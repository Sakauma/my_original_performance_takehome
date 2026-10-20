# Original Performance Take-Home：880 cycles

本仓库保存 [Original Performance Take-Home](https://github.com/anthropics/original_performance_takehome) 的 880 周期实现。提交文件为 `perf_takehome.py`，通过单核 SIMD/VLIW 指令完成二叉树遍历和哈希计算。

## 结果

| 指标 | 结果 |
| --- | ---: |
| 执行周期 | **880 cycles** |
| 官方基线 | 147734 cycles |
| 相对基线加速 | **167.88×** |
| Scratch 使用量 | **1437 / 1536 words** |
| 指令操作总数 | 19815 |
| 官方测试 | **9/9 通过** |

工作负载为树高 10、2047 个节点、256 个输入、16 轮；初始节点索引均为 0，输出为 256 个最终值。机器配置为 1 核、向量长度 8，每周期 ALU/VALU/LOAD/STORE/FLOW 的槽位上限分别为 12/6/2/2/1。

历史成绩：982 → 959 → 913 → 899 → 891 → 881 → **880**。

## 实现

内核直接生成最终指令。`KernelBuilder.build_kernel` 调用 `compile_program`，由 `_Emitter` 把计算放入 880 个指令束。`_SCRATCH_LAYOUT` 记录各项计算的 scratch 地址，`_ISSUE_PLAN` 只记录计算坐标与发射周期；所有运算及源操作数都显式写在语义 emitter 中。提交文件自包含，不依赖实验文件、优化器或指令回放。

| 代码入口 | 职责 |
| --- | --- |
| `_emit_setup` | 读取头部与输入，准备常量和浅层树缓存 |
| `_emit_tree_packets` | 将子树整理为连续的 3 节点、7 节点数据包 |
| `_emit_traversal` | 计算遍历路径、节点地址和预取 |
| `_emit_hash` | 执行六阶段 32 位哈希计算 |
| `_emit_memory_selections` | 通过临时内存完成部分分支选择 |
| `_emit_output` | 完成末尾 XOR 并写回最终值 |

主要优化：

- 将 256 个输入组织成 32 组八路向量，交错执行各组的遍历、访存和哈希计算。
- 缓存浅层节点，将中层子树重排为逆序、XOR 编码的数据包，结合兄弟节点预取减少访存等待。
- 用乘加等价变换合并哈希运算，并将末尾 XOR 延后到后续轮次或输出阶段。
- 用条件写入与向量加载完成部分选择，分摊 FLOW 引擎负载；提前计算末轮分支以缩短依赖链。
- 配合标量、向量指令调度和 scratch 复用，提高各引擎的并行度。
- 重叠部分 packet 加载窗口，利用已消费内存广播四个常量，并在末轮第 29 组中通过 lane 置换与单次标量旋转收集分支数据。

内核将已消费的头部、部分树节点和输入索引区用作临时空间，最终值写回输入值区域。相同进程内的重复构建会复用编译结果缓存。

## 复现

使用 Python 3.12，在终端执行：

```bash
git clone https://github.com/Sakauma/my_original_performance_takehome.git
git clone https://github.com/anthropics/original_performance_takehome.git
cd original_performance_takehome
git checkout 5452f74bd977807ac2e74f3d29432b9df6f25197
cp ../my_original_performance_takehome/perf_takehome.py ./perf_takehome.py
python tests/submission_tests.py
```

在 Windows PowerShell 中，复制源码的命令为：

```powershell
Copy-Item ..\my_original_performance_takehome\perf_takehome.py .\perf_takehome.py
```

预期结果为 9 项测试通过，九次周期输出均为 `880`。

## 验证记录

2026-10-20，在未修改官方测试、冻结参考实现及机器配置的前提下验证：

- 正式 `perf_takehome.py` 通过官方测试 **9/9** 与补充验证 **277/277**，均为 **880 cycles**，scratch 为 **1437 / 1536 words**。
- 原生 emitter 的全部 **19815** 条操作按周期及引擎与已验证候选逐条一致；两张静态表只包含数值布局和调度数据。

2026-10-08，在 Windows、Python 3.12.14 和上述官方提交上验证：

- 官方测试 **9/9 通过**，九次测量均为 **881 cycles**；测试报告用时约 **0.56 秒**。
- 首次内核构建约 **34 毫秒**。
- 补充验证 **277/277 组通过**，覆盖 32 位和 31 位随机值、边界值组合、边界值循环、逐位模式及相同输入值批次；最终值均与冻结参考实现一致，周期数均为 881。
- 指令槽容量、scratch 地址范围、同周期 scratch 写入、运行时内存边界和内存写入冲突检查均通过。
