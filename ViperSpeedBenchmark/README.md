# ViperSpeedBenchmark

"同内核多实现测速 + 组间差值核验" 基准测试工程(2026-09 修复版)。

在同一数学内核上对比 **Viper-SIMD / Viper-Scalar(同一 exe 的"向量化 vs 未向量化"对照) /
手写 C / 手写 C++ / Python** 五个实现，
按"进程内多次 run = 一组 × 5 组(默认; 首组仅汇报不计入统计)"记录原始数据、
做组间差值核验，每次 run 的运算量统一为 **≈ 2.147e9 flops**。

## 1. 被测内核与负载(四语言一致)

```text
double 向量, LEN = 16384 (2^14, 4 的倍数)
输入 a / b / c 用固定种子伪随机(LCG, minstd 参数)填充一次, 三语言逐位一致
out[i] = a[i] * b[i] + c[i]        # 元素独立, 单趟融合乘加; 每元素 2 次浮点运算
每次计时 run = ITERS = 65536 趟内核  ->  总 flops = LEN*ITERS*2 = 2147483648 ≈ 2.147e9
```

- **工作集刻意压在私有 L2 之内(512 KiB)**: `a/b/c/out` 四份共 512 KiB, 本机
  (AMD Ryzen AI 7 H 350)每核 **1 MiB 私有 L2**, 所以访存不落到共享 L3 / DRAM ——
  不再被其它进程抢带宽, 这是消除"零星高时延尖峰"的关键。`LEN × ITERS × 2` 恒为
  ≈2.147e9 flops/run, 所以 GFLOPS 与旧版(DRAM 级工作集)可直接对比。
  再往下压到 L1(需 `LEN ≤ 1024`)会让每趟的内核调用/派发开销占比飙升, 对
  Python+numpy 尤其致命(实测每趟固定开销约 0.4 µs)。

- **LCG 公式**(所有实现逐位一致):
  `state = fmod(state * 48271.0, 2147483647.0);  val = state / 2147483647.0`
  种子固定 `12345.0`。C 用 `math.h::fmod`, Viper 用 `vipermath.fmod`, Python 用
  `math.fmod`(取值范围内逐位一致)。填充顺序统一为逐元素 `a[i], b[i], c[i]`
  依次消耗同一条 LCG 流。
- **Viper = 单趟真硬件 FMA**: 在 `vipersimd` 新增批量内核
  `simd_fma_array(a, b, c_val, out, n)`(仿照 `simd_mul_array` 的 ×8 双缓冲 AVX2
  ymm 循环, 每大迭代 16 路 load 在飞后 8 条 `vfmadd213pd` 一次完成 `out=a*b+c`,
  加数 c 直接用内存操作数) —— **不是**旧版 `mul→tmp→add` 的两遍写法, 无中间数组、
  无 1.5 倍内存流量(见 `includes/vipersimd.vp`)。
- **Viper 原生对照内核(SIMD 加速比的基准线)**: 每个操作都有一对"SIMD / 未向量化"
  实现, 两侧用的是对应宽度的同一条 IEEE 指令, 因此逐元素结果**逐位一致**, CHECK 可直接对照:

  | mode | 操作 | SIMD 实现 | 原生(标量)实现 | 标量指令 |
  | --- | --- | --- | --- | --- |
  | 0 / 1 | FMA `a*b+c` | `simd_fma_array` | `scalar_fma_array` | `vfmadd213sd` |
  | 2 / 3 | ADD `a+b` | `simd_add_array` | `scalar_add_array` | `vaddsd` |
  | 4 / 5 | MUL `a*b` | `simd_mul_array` | `scalar_mul_array` | `vmulsd` |
  | 6 / 7 | SQRT `sqrt(a)` | `simd_sqrt_array` | `scalar_sqrt_array` | `vsqrtsd` |

  同一个 `app.exe` 靠命令行第 4 个参数 `mode` 选内核, 除"是否向量化"外没有任何其它
  差异; `run_bench.py` 把它们记成 `viper* / viper_scalar*` 两组数据, 供
  `plot_bench.py` 图1 画 ADD/MUL/SQRT 的延迟对比与加速比。
  > ADD/MUL/SQRT 每元素只有 1 次浮点运算(FMA 是 2 次), 所以它们的单 run flops 也
  > 如实按 1 计, 耗时约为 FMA 的一半 —— 图1 比的是"同操作下原生 vs SIMD"的倍数,
  > 不受这个差异影响。
- **CHECK 舍入口径(2026-09 起 C/C++ 也开 FMA)**: C/C++ 与 Viper 都走
  **单趟真硬件 FMA** —— C/C++ 编译加 `-mfma -ffp-contract=fast`, 让
  `a[i]*b[i]+c[i]` 收缩成一条 `vfmadd`, 逐元素结果是**单次舍入**, 与 Viper 的
  `simd_fma_array` 同口径, 两者 CHECK 应当逐位一致; Python+numpy 仍是
  `multiply→tmp→add` **两次舍入**, 与前三者可能差 1 ulp。各实现**内部自洽**
  (每次进程 CHECK 恒定), 但 numpy 与原生实现**不逐位相等**属预期。
- **线程绑定(每个实现必做)**: 进程入口调 `SetThreadAffinityMask` 把自身绑到
  **第一个可用逻辑核**(从 `1<<0` 逐个试, 第一个被系统接受的位), 并以
  `AFFINITY <core>` 行回报; `-1` = 绑定失败(不致命)。各实现会绑到同一个核,
  消除跨核迁移、大小核异构与缓存失效带来的抖动, 横向对比才有意义。
- **双计时(每次 run 同时采集两个数)**:

  | 字段 | 来源 | 特性 |
  | --- | --- | --- |
  | `wall_us` | `QueryPerformanceCounter` | 墙钟; **含**被抢占 / 被降频的时间, 对调度敏感 |
  | `cpu_us` | `QueryThreadCycleTime` | 只计本线程真正执行过的周期; **不含**被抢占时间, 对降频也免疫 |

  TSC 频率由各实现**入口自行标定**: 绑核之后忙等 30ms 数周期, 重复 **3 次取最大**
  (取最大是为了压掉"标定窗口内被抢占导致 hz 偏小"的系统性正偏置 —— 该偏置会让
  `cpu_us` 偶尔比 `wall_us` 还大, 看着像 bug)。
- **为什么不用 `GetThreadTimes`**: 它的 CPU 记账粒度跟随调度时钟节拍, 本机实测
  硬地板 **15.625 ms** —— 一个 ~37ms 的 run 只能被记成 2 或 3 个节拍
  (31250 / 46875 µs), 误差 **±40%**, 连"CPU 时间 > 墙钟时间"这种物理上不可能
  的结果都会出现。已验证 `timeBeginPeriod(1)` 在 Win10 2004+ **无法**改善该粒度
  (加完后值仍是 15625 的整数倍)。`QueryThreadCycleTime` 的粒度是周期级。
- **统一输出协议**(每个可执行进程开头: `CFG` / `AFFINITY` / `TSC` 各一行; 之后每次 run 一行):
  ```text
  CFG      <len> <iters>
  AFFINITY <core>                   # -1 = 绑定失败
  TSC      <hz>                     # 入口标定的 TSC 频率; 0 = 标定失败(该实现周期计数未推进)
  RESULT   <wall_us> <cpu_us> <flops>   # flops = (LEN*ITERS*2) / 墙钟秒
  CHECK    <sum>                    # 全部结束后, 最后一次 run 的 out 升序累加
  ```
- **线程优先级**: 四个实现绑核后统一调 `SetThreadPriority(THREAD_PRIORITY_HIGHEST)`,
  压低背景进程抢占造成的"零星高时延尖峰"。
- **Viper 侧为什么用 `OpenThread` 而不是 `GetCurrentThread()`**: 后者返回伪句柄
  `(HANDLE)-2`(负指针)。在 C / Python 里它只是一段裸位模式, 可以原样传下去;
  但 Viper 的 `t.CPtr` 语义下传递不可靠 —— 实测 `QueryThreadCycleTime` 只写回
  约 `0.230 µs`(≈ 一次调用开销的周期数), 即计数器根本没被读出来。
  改用 `GetCurrentThreadId()` + `OpenThread()` 拿真句柄, 全程 `t.CSizeT` 表达,
  彻底绕开指针/NULL 语义。

## 2. 修复内容(相对旧版 bench_report 暴露的问题)

| 旧问题 | 本次修复 |
| --- | --- |
| 噪声过大 → 差值核验全 FAIL | 旧版"每次 run 一次冷进程"; 现每组=**一次进程启动**, 进程内先 **2 轮满负荷热身**(每轮 = ITERS 趟 + 读回 out 校验)再连续计时 N 次, 消除冷页/启动/频率爬升噪声 |
| Python 段缺失 | run_bench.py 默认各实现都跑(`py` 用 `sys.executable bench_py.py <runs>`), 任何启动/解析异常打印完整 traceback |
| vipersimd 力度不足 | 弃用 `simd_mul_array`+`simd_add_array` 两遍, 新增/调用 `simd_fma_array` 单趟 FMA(×8 ymm 双缓冲, 3 输入 1 输出全走寄存器/访存流水) |
| 运算量太小 | LEN=16384 × ITERS=65536 × 2 ≈ **2.147e9 flops/次 run**(旧版约 0.5e6) |
| 访存打到共享 L3 / DRAM, 被其它进程抢带宽 → 零星高时延尖峰 | 把工作集压到 **512 KiB**(四份缓冲合计), 稳居每核私有 L2 之内; 同时绑核 + 提线程优先级 |
| 跨核迁移 / 调度噪声无法定位 | 每个实现入口 **绑定固定逻辑核**(`SetThreadAffinityMask`), 并**双计时**: 墙钟(QPC) + 线程 CPU(`QueryThreadCycleTime`)。二者差额即被抢占/被降频吃掉的部分 |
| 离群样本把组均值拉偏 | 组内**先按中位数上界裁剪**(`--trim-ratio`, 默认 1.2)再统计; 并提供 `--stat median/geomean` 应对双峰分布 |
| C/C++ 与 Viper 舍入口径不一致 | C/C++ 放开 FMA 收缩(`-mfma -ffp-contract=fast`), 与 Viper 的 `simd_fma_array` 同为单趟硬件 FMA 单次舍入 |

## 3. 文件清单

| 文件 | 说明 |
| --- | --- |
| `project.vpj` | Viper 工程配置(产物 `__vpcbuild__/output/app.exe`) |
| `App/main.vp` | Viper 实现: 8 个内核(FMA/ADD/MUL/SQRT × SIMD/原生) + 线程绑定 + QPC/周期双计时 + 进程内多 run; 自解析命令行 `[runs] [len] [iters] [mode]`(GetCommandLineA, 不依赖 sys/memhub) |
| `bench_c.c` | 手写 C: 裸数组 + **全指针 `__restrict`**(fill/内核/校验和/local 指针都加) + noinline, `-O2 -mavx2 -mfma -ffp-contract=fast` 自动向量化单趟 FMA |
| `bench_cpp.cpp` | 手写 C++: `std::vector` + `__restrict`, 同口径(`-O2 -mavx2 -mfma -ffp-contract=fast`) |
| `bench_py.py` | Python: 优先 numpy(就地 ufunc **两次舍入**, 与 C/C++/Viper 的 FMA 单次舍入可能差 1 ulp); 无 numpy 自动纯 Python 降规模回退(趟数=8, 如实打印 CFG)。线程绑定与线程 CPU 时间走 ctypes 调 kernel32 |
| `run_bench.py` | 总编排: 每组一次子进程 + 解析 `CFG/MODE/AFFINITY/TSC/RESULT/CHECK` + 离群裁剪 + 分组统计与差值核验 + 多尺寸扫描 + 控制台/`bench_report.txt`/CSV + `--flops-check`。Viper 侧 8 个内核变体(`viper`/`viper_scalar`/`viper_add`/`viper_scalar_add`/`viper_mul`/`viper_scalar_mul`/`viper_sqrt`/`viper_scalar_sqrt`, 同一个 exe 只差 `mode`) + `c` / `cpp` / `py` |
| `plot_bench.py` | 读 run_bench 导出的 CSV 出 4 张图(matplotlib): ①左=ADD/MUL/SQRT 原生 vs SIMD 延迟、右=各操作加速比 ②各实现耗时折线(叠加每次 run 实测点) ③Overall Mean 耗时对比 ④Measured GFLOPS 对比; 另附控制台文字小结 |
| `bench_report.txt` | 最近一次运行日志(由 run_bench.py 生成) |
| `../../includes/vipersimd.vp` | 新增批量内核 `simd_fma_array`(单趟融合乘加, ×8 ymm 双缓冲) 与 4 个原生对照内核 `scalar_fma_array` / `scalar_add_array` / `scalar_mul_array` / `scalar_sqrt_array`(逐元素标量指令, 与对应 SIMD 版逐位同结果) |

## 4. 前置条件

- Viper 工具链: 可构建 `project.vpj`(见 §5.1)。
- C/C++ 工具链(默认): `D:/msys64/mingw64/bin/clang.exe` 与 `clang++.exe`;
  其它目录用 `--cc-dir` 指定。
- Python 3.8+(numpy **可选**, 有则全规模 2.147e9 flops; 无则纯 Python 回退)。
- CPU 需支持 **AVX2 + FMA**(vipersimd 与 C/C++ 的 `-mavx2 -mfma` 均使用 YMM/FMA)。

## 5. 如何构建与运行

### 5.1 构建 Viper 部分(一次性)

在本工程目录(`ViperSpeedBenchmark/`)用你惯用的 tpv 方式构建 `project.vpj`,
产物应为 `__vpcbuild__/output/app.exe`。若未构建, `run_bench.py` 会提示并**跳过
全部 Viper 内核变体** 继续其它实现。

> **改了 `App/main.vp` 或 `includes/vipersimd.vp` 后必须重建**, 否则任一 Viper
> 内核变体都会因为产物回报的 `MODE` 与期望不符(旧 exe 不认命令行第 4 个参数、
> 或没有新内核)而显式报错 —— 这是有意为之: 宁可失败, 也不让某个内核的数据被
> 另一个内核冒充, 污染 SIMD/原生 对照。

### 5.2 运行完整基准(默认: 8 个 Viper 内核变体 + C/C++/Python, 共 11 个; 每组 = 一次进程, 进程内 10 次 run)

```bat
python run_bench.py
```

常用参数:

```bat
:: 只跑部分实现
python run_bench.py --impl c,cpp
:: Viper 的 SIMD vs 原生 对照(同一个 app.exe, 只差命令行 mode 参数)
python run_bench.py --impl viper,viper_scalar --sizes 1024,4096,16384
:: 只要图1 的内核成对数据(默认已含它们, 这里只是把范围收窄)
python run_bench.py --impl viper_add,viper_scalar_add,viper_mul,viper_scalar_mul,viper_sqrt,viper_scalar_sqrt
:: 只要图2/3/4 的跨语言对比(跳过 6 个 ADD/MUL/SQRT 变体)
python run_bench.py --impl viper,viper_scalar,c,cpp,py
:: 快速冒烟(减少组数/每组次数; 每次进程仍做 2 轮满热身)
python run_bench.py --impl c,cpp,py --groups 2 --runs-per-group 3
:: 自定义容差 / 导出原始数据 CSV / 启用 flops 核验
python run_bench.py --tol 0.05 --csv out.csv
python run_bench.py --flops-check
:: 多尺寸扫描: LEN 决定工作集(四份缓冲共 LEN*32 字节), ITERS 自动配平
python run_bench.py --sizes 1024,4096,16384,65536,262144
:: 抗噪口径: 用线程 CPU 时间当主指标 + 中位数 + 裁剪
python run_bench.py --metric cpu --stat median --trim-ratio 1.2
:: 关闭离群裁剪(回到"全部样本计入")
python run_bench.py --trim-ratio 0
:: Python 模式强制
python run_bench.py --impl py --py-mode pure
python run_bench.py --impl py --py-mode numpy
:: 自定义 clang 目录 / 自定义 viper 产物
python run_bench.py --cc-dir C:/msys64/mingw64/bin
python run_bench.py --viper-exe D:/path/to/app.exe
```

C/C++ 源码更新(或编译选项变化)后会自动重编到 `.native_build/`。

**单 run 期望时长**: 每次 run ≈ 2.147e9 flops, 工作集 512 KiB 常驻 L2。实测吞吐
取决于机器: 原生(C/C++/Viper, 单线程 AVX2+FMA)大致 **几十 ms** 量级;
Python+numpy 与之同规模、通常慢 1~3 倍; 纯 Python 回退
(无 numpy)会把趟数自动降为 8, 单 run 约 0.2~0.5 s。**完整默认跑 11 个实现 × 10 进程,
墙钟约 1~5 分钟**, 快速验证请用 §5.2 冒烟参数。

### 5.3 输出与日志

- 控制台实时输出; 结束后全文写入 `bench_report.txt`。
- `--csv out.csv`: 逐样本原始行, 列 =
  `impl, len, iters, group, run, wall_us, cpu_us, flops, check, trimmed`
  (`trimmed=1` 表示该样本被离群裁剪剔除, 未计入该组的组统计量)。
- `--flops-check`: 核对各实现 CFG 单 run 总运算量是否 ≈ 2.147e9, 并逐 RESULT 行
  验证 `flops` 字段与 `us` 换算一致(`flops = CFG_total / 秒`);
  对纯 Python 降规模配置给出提示。注意 `flops` 是**吞吐**, 只有单 run 时长恰 ≈1 s
  时数值才 ≈2e9, 报告会同时给出实测 GFLOPS。

## 6. 多尺寸扫描与出图

### 6.1 `--sizes`: 一次跑出"吞吐 vs 工作集"曲线

`--sizes` 接受逗号分隔的 `LEN` 列表。**只调 LEN, ITERS 自动配平**(`ITERS = 2^30 / LEN`),
所以每次 run 的运算量恒为 ≈2.147e9 flops, 各尺寸的 GFLOPS 可直接横向比较。
工作集 = `a/b/c/out` 四份合计 = `LEN × 32` 字节:

| LEN | ITERS | 工作集 | 落在 |
| --- | --- | --- | --- |
| `1024` | 1048576 | 32 KiB | L1D(本机 48 KiB) |
| `4096` | 262144 | 128 KiB | L2 |
| `16384` | 65536 | 512 KiB | **L2(默认, 本机每核 1 MiB)** |
| `65536` | 16384 | 2 MiB | L3(本机 16 MB 共享) |
| `262144` | 4096 | 8 MiB | L3 / 边缘 |
| `1048576` | 1024 | 32 MiB | DRAM |

`LEN` 必须是 4 的倍数(内核按 4-lane ymm 分批), 且必须让 `2^30 / LEN` 整除;
不满足时 run_bench.py **直接报错退出**, 不做静默修正。

> 想看"纯算力"(SIMD 能力)看小尺寸那一端, 想看"访存带宽"看大尺寸那一端。
> 本机四个实现应该都会在 **512 KiB~2 MiB 附近出现一个峰**(私有 L2 内),
> 再加大就掉到共享 L3/DRAM 上 —— 曲线拐点就是这台机器各级缓存的边界。

### 6.2 自动导 CSV

run_bench.py **默认自动导出** CSV(带时间戳, 如 `bench_20260910_203105.csv`),
不用再加 `--csv`。要指定文件名用 `--csv name.csv`, 完全不写用 `--no-csv`。

CSV 列: `impl, len, iters, group, run, wall_us, cpu_us, flops, check, trimmed`
(`trimmed=1` 表示该样本被离群裁剪剔除)。

### 6.3 `plot_bench.py`: 从 CSV 出 4 张图

```bat
python plot_bench.py                                    :: 自动取最新的 bench_*.csv
python plot_bench.py bench_20260910_203105.csv
python plot_bench.py --metric cpu
python plot_bench.py bench_xxx.csv --out-dir figs --prefix run1
python plot_bench.py --simd-impl viper --scalar-impl viper_scalar
```

产出 4 张 PNG(默认与 CSV 同目录, 文件名前缀 = CSV 名):

1. `<前缀>_1_simd_vs_scalar.png` —— **左右双面板**:
   - **左**: 每种操作(ADD / MUL / SQRT)一组柱, 组内左=原生(标量)、右=**ViperSIMD**,
     柱顶标该操作的每 run 均值(us)
   - **右**: 同一批操作的 **SIMD 加速比**(原生延迟 ÷ SIMD 延迟), 柱顶标 `xN.NN`,
     并画一条 `x1 = no gain` 参考线
   **前提**: CSV 里要同时有这三种操作的 原生+SIMD 成对数据(见 §5.2 的 "图1 ... 内核成对数据");
   缺哪一对就少画哪一根, 全缺则整张图跳过并提示。只取 CSV 里**最小的那个 LEN**。
2. `<前缀>_2_exec_time.png` —— **Execution Time(microseconds)**:
   每实现一条均值折线, 并把**每组内每次 run 的实测点**画成同色散点(各实现横向错开
   以免重叠)。多尺寸时横轴 = LEN(折线即"工作集-耗时曲线"); **只跑了 1 个 LEN 时
   横轴自动改用组号**(每组 = 一次进程内的多次 run), 否则单点连不成折线。
   动态范围 >30 倍时纵轴自动转对数(`--y-scale log|linear`)。
3. `<前缀>_3_overall_mean.png` —— **Overall Mean Execution Time**:
   每实现一根柱(全部尺寸、全部组有效样本合并平均), 误差棒 = 样本标准差,
   柱顶标绝对耗时, 柱身标"相对最快"倍数。
4. `<前缀>_4_gflops.png` —— **Measured GFLOPS Throughput**:
   每实现一根柱(`GFLOPS = LEN×ITERS×2 ÷ 单 run 秒数`, 逐样本算后取平均),
   误差棒 = 样本标准差, 柱身标"相对最快"倍数。

三张柱/线图都跟随 `--metric`(`wall` 默认 QPC 墙钟 / `cpu` 线程 CPU 时间),
且与 run_bench.py 报告同口径: CSV 里 `trimmed=1` 的离群样本不计入。

同时在控制台打一份文字小结(`impl / LEN / 工作集 / 每run耗时 / GFLOPS / CPU比 / trimmed`)。

> 依赖 `matplotlib` + `numpy`。**图内文字一律用英文**(短句 + 多行换行, 避免长句把标题
> 撑爆); 控制台小结仍为中文。字体已显式配置 `Microsoft YaHei`, 万一要出中文图也不会变方块。

---

## 7. 统计口径与差值核验(默认容差 5%)

每个实现(`impl`)每次 run 产出 **两个**原始值(`wall_us` 与 `cpu_us`); **一组 = 一次
进程启动**内的 `runs_per_group`(=10)次连续计时。

### 7.1 主指标、离群裁剪、统计量

- **主指标 `--metric`**(默认 `wall`): 决定组均值 / 总体 / 差值核验用哪个字段。
  另一指标同时采集, 在报告的"对照"行与最终汇总里一并给出, 也都写进 CSV。
  被抢占较多的机器上建议试试 `--metric cpu`。
- **离群裁剪 `--trim-ratio`**(默认 `1.2`): 每组**先**按组内中位数剔上界离群点 ——
  只保留 `值 <= 组内中位数 × 1.2` 的样本, 再算该组统计量。`<=0` 关闭裁剪。
- **统计量 `--stat`**(默认 `mean`): `mean`(裁剪后算术平均) / `median` / `geomean`。
  报告里三个值**同时列出**, 不必重跑。**双峰分布**(一半样本被降频或被抢占)下
  `median` 的抗噪能力远强于算术平均。

### 7.2 各统计量的定义

对每个实现:

- **组统计值**: 组内 10 次原始值**先经离群裁剪**, 再按 `--stat` 取统计量
  (默认 = 裁剪后算术平均)。报告同时列出该组的 均值 / 中位数 / 几何均值。
- **组内差(max-min)**: 裁剪后保留样本的最大值 - 最小值(主要反映进程内稳态抖动)。
- **总体统计值**: 该实现全部 `统计组数 × runs_per_group` 个**裁剪后**样本按 `--stat`
  统计(默认排除首组后)。
- **样本标准差 / 最大偏差**: 裁剪后样本的 n-1 无偏标准差; `|样本-总体|` 最大者。
- **组 vs 总体差值**: `组统计值 - 总体统计值`(绝对 us 与相对百分比)。
- **相邻组差值**: `组统计值[g+1] - 组统计值[g]`(绝对 us 与相对百分比)。
- **差值核验**: 每组 `|组统计值-总体|/总体 <= tol` 且 `|相邻组差值|/总体 <= tol`
  → `PASS`, 否则列出违规组 → `FAIL`。默认 tol = 0.05, `--tol` 可调。

> **首组效应规避(默认)**: 组1(第一次进程启动)在报告中**仍逐组/逐 run 展示**,
> 但**不计入**总体均值、样本标准差、最大偏差与差值核验(含组1->组2 相邻差),
> 用于剔除冷启动/频率爬升阶段的偏差。需要把组1纳入统计时加 `--include-first-group`。

> 扰动缓解说明: (1) 每个进程入口 **绑定到固定逻辑核**(`AFFINITY`), 消除跨核迁移与
> 大小核异构; (2) 每个进程内先 **2 轮满负荷热身**(每轮 = ITERS 趟内核 + 读回 out
> 求校验和), 缓存/TLB/频率进入稳态后才开始计时; (3) 组内多次 run 取自**同一进程**,
> 没有冷启动开销; (4) 每次 run 同时给出**墙钟**与**线程 CPU 时间**, 二者的差额就是
> 被抢占 / 被降频吃掉的部分 —— 若墙钟抖动大而 CPU 时间稳, 说明是调度噪声而非代码问题;
> (5) 组内离群点先按中位数上界裁掉, 双峰分布下可用 `--stat median` 进一步抗噪;
> (6) 尺寸扫描时 **LEN×ITERS 恒定**(单 run 运算量恒为 ≈2.147e9 flops), 只让工作集变化,
> 组间只剩真实抖动。

## 8. 防止"优化器把循环删掉"与已知注意点

- **C/C++**: 内核 noinline + 全指针 `__restrict`(C 与 C++ 统一拼写, 不再一个用
  C99 `restrict` 一个用 `__restrict`); 每轮热身与每次 run **之后**
  升序读回 `out` 求和(计时区外), 使计时区内每条写可观察、阻止跨 run 写折叠;
  未开 fast-math, 但显式开 `-ffp-contract=fast` + `-mfma`, 允许收缩成单趟 FMA
  (与 Viper 同口径、同一舍入次数)。要复现旧的"两次舍入"口径做 A/B, 把
  `run_bench.py` 里的 `NATIVE_FLAGS` 改回 `-ffp-contract=off` 并去掉 `-mfma` 即可。
- **Viper**: `simd_fma_array` 内联汇编自带 memory clobber, 计时区外同样读回校验。
- **Python**: numpy 就地 ufunc / 纯 Python 循环都不存在被删问题; 纯 Python 回退按
  实际趟数打印 CFG(≠2e9 配置), `--flops-check` 会给出提示。
- Viper 与 C/C++ 现在同为单次舍入 FMA, CHECK 应当逐位一致; numpy 是两次舍入, 与它们
  可能差 1 ulp(见 §1), 跨实现比较 CHECK 数值前先确认这一点。
- **线程 CPU 时间(`cpu_us`)的适用边界**: 它由 `QueryThreadCycleTime` 的周期数除以
  **入口标定的 TSC 频率**得到。现代 x86 的 TSC 是 invariant 的(与核心实际频率无关),
  所以该值就是"本线程真正跑过的时间"; 但在极老的非 invariant TSC 机器上, 周期数会
  跟着核心频率走, 换算出的 `cpu_us` 会失真 —— 那种机器上请以 `--metric wall` 为准。
  另外标定本身有约 0.1% 的残差(已用三次取最大压掉大部分), 不影响各实现之间的横向对比。
- 建议空闲机器上跑完整基准; CPU 需支持 AVX2+FMA(见 §4)。
