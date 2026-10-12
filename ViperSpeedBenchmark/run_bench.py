# -*- coding: utf-8 -*-
"""
run_bench.py - "同内核多实现测速 + 组间差值核验" 总编排脚本
(ViperSpeedBenchmark, 2026-09 抗噪强化版)

噪声治理设计:
  * 每组(默认 5 组)只启动【一次】子进程: <exe> <runs_per_group=10>。
    该进程内先做 2 轮满负荷热身, 再连续计时并打印 10 行 RESULT ——
    组原始数据来自同进程稳态运行, 消除了旧版"每次 run 一次冷进程"的
    冷页/TLB/启动/频率爬升噪声。
  * 【线程绑定】每个实现在进程入口调 SetThreadAffinityMask 把自身绑到
    第一个可用逻辑核(结果以 AFFINITY <core> 行回报), 消除跨核迁移、
    大小核异构与缓存失效带来的抖动。绑定失败回报 -1, 不致命。
  * 【双计时】每次 run 同时采集:
        wall = QPC 墙钟微秒        (含被抢占/被降频的时间, 对调度敏感)
        cpu  = QueryThreadCycleTime 线程 CPU 微秒
                                   (只计本线程真正执行过的周期, 对抢占/降频
                                    不敏感; TSC 频率由各实现入口自行标定)
    两种都进 RESULT 行、都做统计、都写 CSV; --metric 选谁当"主指标"决定
    组均值/总体/差值核验与 PASS/FAIL。
    注: 不用 GetThreadTimes —— 它的记账粒度跟随调度时钟节拍(本机实测硬地板
        15.625ms), 几十毫秒的 run 误差 ±40%, 且 Win10 2004+ 起无法用
        timeBeginPeriod 改善。
  * 【离群裁剪】每组先按中位数上界裁掉离群点(默认 保留 <= 中位数×1.2 的样本),
    再算组统计量。裁剪系数由 --trim-ratio 控制, <=0 关闭。
  * 【统计量可选】--stat mean(默认, 裁剪后算术平均) / median / geomean。
    双峰分布(一半样本被降频)时 median 的抗噪能力远强于算术平均。

  * 【多尺寸扫描】--sizes 接受逗号分隔的 LEN 列表; 只调 LEN, ITERS 自动配平
    (ITERS = 2^30 / LEN) 使单 run 恒为 ≈2.147e9 flops, 于是 GFLOPS 跨尺寸可比。
    LEN 决定工作集(四份缓冲共 LEN*32 字节): 1024=L1 / 16384=L2 / 262144 以上=L3-DRAM。
  * 【自动导 CSV】默认写成带时间戳的 bench_<YYYYmmdd_HHMMSS>.csv (--csv 指定名字,
    --no-csv 关闭), 可直接喂给 plot_bench.py 出图。

每个可执行统一协议(进程开头一次):
    CFG      <len> <iters>
    AFFINITY <core>                  # -1 = 绑定失败
    TSC      <hz>                    # 入口标定出的 TSC 频率; 0 = 标定失败
    MODE     <0|1>                   # 仅 Viper 输出: 0=simd_fma_array / 1=scalar_fma_array
之后每次 run 一行:
    RESULT   <wall_us> <cpu_us> <flops>
全部结束后:
    CHECK    <sum>

实现集合(全部 Viper 变体共用同一个 app.exe, 只差命令行第 4 个参数 mode):
    viper / viper_scalar            FMA  (out = a*b + c)     mode 0 / 1
    viper_add / viper_scalar_add    ADD  (out = a + b)       mode 2 / 3
    viper_mul / viper_scalar_mul    MUL  (out = a * b)       mode 4 / 5
    viper_sqrt / viper_scalar_sqrt  SQRT (out = sqrt(a))     mode 6 / 7
    c / cpp                         手写 C/C++, -O2 -mavx2 -mfma -ffp-contract=fast
    py                              Python numpy(就地 ufunc); 无 numpy 则纯 Python 降规模回退
  每对 `<op>` / `scalar_<op>` 是同一数学内核的"向量化 vs 未向量化"对照 —— 标量版用
  对应标量指令(vaddsd/vmulsd/vsqrtsd/vfmadd213sd), 逐元素结果与 SIMD 版逐位一致,
  CHECK 可直接对照; plot_bench.py 图1 就是用它们算 ADD/MUL/SQRT 的 SIMD 加速比。

统计(每实现): 组均值 / 组内 max-min / 组-总体 / 相邻组差 / 总体均值 /
样本标准差 / 最大偏差 / 裁剪剔除数; 差值核验规则同前(容差默认 0.05)。

首组处理: 首组(组1)默认【仅汇报, 不计入】总体均值/样本标准差/差值核验
(规避首组效应: 第一次进程启动的冷启动/频率爬升)。需要计入加 --include-first-group。

用法示例:
    python run_bench.py
    python run_bench.py --impl c,cpp,py --groups 3 --runs-per-group 5
    python run_bench.py --impl viper,c,cpp,py --tol 0.05 --csv out.csv --flops-check
    python run_bench.py --metric cpu --stat median                # 抗噪口径
    python run_bench.py --trim-ratio 0 --stat mean                # 关闭裁剪
    python run_bench.py --impl py --py-mode pure                  # 强制纯 Python 回退
"""
import argparse
import csv
import datetime
import math
import os
import re
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CC_DIR_DEFAULT = "D:/msys64/mingw64/bin"

DEFAULT_LEN = 16384
DEFAULT_SIZES = "16384"
TOTAL_FLOPS_TARGET = 1 << 31          # 2147483648 ≈ 2.147e9, 每次 run 恒定的运算量
EXPECT_TOTAL_FLOPS = float(TOTAL_FLOPS_TARGET)

CFG_RE = re.compile(r"^\s*CFG\s+(\d+)\s+(\d+)")
AFFINITY_RE = re.compile(r"^\s*AFFINITY\s+(-?\d+)")
# 注意: 内层用非捕获分组, 使 RESULT 的第 1/2/3 个数字分别落在 group(1)/group(2)/group(3)
NUM = r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?"
RESULT_RE = re.compile(rf"^\s*RESULT\s+({NUM})\s+({NUM})\s+({NUM})")
CHECK_RE = re.compile(rf"^\s*CHECK\s+({NUM})")
TSC_RE = re.compile(rf"^\s*TSC\s+({NUM})")
# Viper 专有: MODE <0|1> —— 0=simd_fma_array / 1=scalar_fma_array
MODE_RE = re.compile(r"^\s*MODE\s+(-?\d+)")

METRIC_IDX = {"wall": 0, "cpu": 1}
METRIC_LABEL = {"wall": "墙钟(QPC)", "cpu": "线程CPU(QueryThreadCycleTime)"}
STAT_LABEL = {"mean": "算术平均(裁剪后)", "median": "中位数", "geomean": "几何均值"}


def parse_process_output(text, expected_runs, impl):
    """解析单个子进程 stdout -> dict。

    协议: 1 行 CFG + 可选 1 行 AFFINITY + expected_runs 行 RESULT + 1 行 CHECK。
    RESULT = <wall_us> <cpu_us> <flops>。
    解析失败抛 ValueError(由调用方转成带 traceback 的失败记录)。
    """
    cfg = None
    affinity = None
    tsc_hz = None
    kernel_mode = None     # Viper 专有: 0=SIMD / 1=Scalar; 其它实现无此行
    samples = []   # (wall_us, cpu_us, flops)
    checks = []
    for line in text.splitlines():
        m = RESULT_RE.match(line)
        if m:
            samples.append((float(m.group(1)), float(m.group(2)), float(m.group(3))))
            continue
        m = CHECK_RE.match(line)
        if m:
            checks.append(float(m.group(1)))
            continue
        m = AFFINITY_RE.match(line)
        if m and affinity is None:
            affinity = int(m.group(1))
            continue
        m = TSC_RE.match(line)
        if m and tsc_hz is None:
            tsc_hz = float(m.group(1))
            continue
        m = MODE_RE.match(line)
        if m and kernel_mode is None:
            kernel_mode = int(m.group(1))
            continue
        m = CFG_RE.match(line)
        if m and cfg is None:
            cfg = (int(m.group(1)), int(m.group(2)))
    if cfg is None:
        raise ValueError(
            f"[{impl}] stdout 未找到 CFG <len> <iters> 行。stdout 前 20 行:\n"
            + "\n".join(text.splitlines()[:20]))
    if len(samples) != expected_runs:
        raise ValueError(
            f"[{impl}] 期望 {expected_runs} 行 RESULT, 实际 {len(samples)} 行"
            f"(CFG={cfg[0]}x{cfg[1]})。stdout 前 40 行:\n"
            + "\n".join(text.splitlines()[:40]))
    if not checks:
        raise ValueError(
            f"[{impl}] stdout 未找到 CHECK 行。stdout 前 40 行:\n"
            + "\n".join(text.splitlines()[:40]))
    return {
        "cfg": cfg,
        "affinity": affinity,
        "tsc_hz": tsc_hz,
        "kernel_mode": kernel_mode,
        "samples": samples,        # list[(wall_us, cpu_us, flops)]
        "check": checks[0],
    }


class BenchError(Exception):
    pass


def sh_ok(cmd, cwd, env=None, timeout=1800):
    """执行命令, 成功返回 CompletedProcess; 失败抛 BenchError(带 stderr)。"""
    try:
        p = subprocess.run(cmd, cwd=str(cwd), capture_output=True,
                           text=True, errors="replace", timeout=timeout, env=env)
    except FileNotFoundError as e:
        raise BenchError(f"找不到可执行文件: {cmd[0]}\n  (请确认路径/工具链已安装)")
    except subprocess.TimeoutExpired:
        raise BenchError(f"命令超时({timeout}s): {cmd}")
    return p


# ------------------------------------------------------------
# 编译 / 运行定义
# ------------------------------------------------------------
# C/C++ 与 Viper 同口径: 允许 FMA 收缩(单趟融合乘加, 单次舍入)。
#   -ffp-contract=fast 让 a[i]*b[i]+c[i] 生成 vfmadd(而不是 mul+add 两次舍入);
#   -mfma 显式打开 FMA 指令集(只给 -mavx2 不保证启用 FMA)。
# 要回到"两次舍入"的旧口径做 A/B, 把 -ffp-contract=fast 换成 =off 并去掉 -mfma 即可。
NATIVE_FLAGS = ["-O2", "-mavx2", "-mfma", "-ffp-contract=fast"]


# Viper 内核实现名 -> (传给 app.exe 的第 4 个参数 mode, 显示名)。
# 全部共用同一个 __vpcbuild__/output/app.exe, 只差这一个 mode 参数 —— 除"跑哪份内核"
# 外没有任何差异, 这是 SIMD vs 原生 对照可信的前提(见 App/main.vp 的 KERNEL_* 常量)。
# 命名规则: `<op>` = SIMD, `scalar_<op>` = 同操作的标量(原生)对照。
VIPER_KERNELS = [
    ("viper",             0, "FMA  SIMD   (vipersimd.simd_fma_array)"),
    ("viper_scalar",      1, "FMA  原生   (vipersimd.scalar_fma_array)"),
    ("viper_add",         2, "ADD  SIMD   (vipersimd.simd_add_array)"),
    ("viper_scalar_add",  3, "ADD  原生   (vipersimd.scalar_add_array)"),
    ("viper_mul",         4, "MUL  SIMD   (vipersimd.simd_mul_array)"),
    ("viper_scalar_mul",  5, "MUL  原生   (vipersimd.scalar_mul_array)"),
    ("viper_sqrt",        6, "SQRT SIMD   (vipersimd.simd_sqrt_array)"),
    ("viper_scalar_sqrt", 7, "SQRT 原生   (vipersimd.scalar_sqrt_array)"),
]
VIPER_MODE = {name: mode for name, mode, _ in VIPER_KERNELS}
VIPER_DISP = {name: disp for name, _, disp in VIPER_KERNELS}

# 默认跑"图1 + 图2/3/4 所需的全部实现": 8 个 Viper 内核变体(FMA/ADD/MUL/SQRT 各一对
# SIMD/原生) + C/C++/Python。图1 需要 ADD/MUL/SQRT 的成对数据, 所以默认就带上 ——
# 否则跑完默认命令图1 会以"缺数据"整张跳过, 体验很差。
# 只要图2/3/4 时, 用 --impl viper,viper_scalar,c,cpp,py 可跳过那 6 个内核变体。
DEFAULT_IMPLS = ",".join([n for n, _, _ in VIPER_KERNELS] + ["c", "cpp", "py"])


def build_native(cc_dir, impl, root):
    """编译 bench_c.c / bench_cpp.cpp 到 .native_build/, 返回 exe 路径。

    源码或编译选项变化时才重编(用 .flags 标记文件记录上次编译选项)。
    """
    build_dir = root / ".native_build"
    build_dir.mkdir(parents=True, exist_ok=True)
    src = root / ("bench_c.c" if impl == "c" else "bench_cpp.cpp")
    exe = build_dir / ("bench_c.exe" if impl == "c" else "bench_cpp.exe")
    cc = cc_dir / ("clang.exe" if impl == "c" else "clang++.exe")
    flags = NATIVE_FLAGS + [
        "-std=c11" if impl == "c" else "-std=c++17",
        "-lmsvcrt", "-lmingwex", "-lkernel32", "-lm",
    ]
    flags_marker = build_dir / (".flags_" + impl)
    need = (not exe.exists()
            or not flags_marker.exists()
            or flags_marker.read_text(encoding="utf-8") != "\n".join(flags)
            or exe.stat().st_mtime < src.stat().st_mtime)
    if not need:
        return exe
    cmd = [str(cc)] + flags + [str(src), "-o", str(exe)]
    p = sh_ok(cmd, root)
    if p.returncode != 0:
        tail = "\n".join(p.stderr.splitlines()[-25:])
        raise BenchError(f"编译失败(exit={p.returncode}): {' '.join(cmd)}\n"
                         f"stderr 尾部:\n{tail}")
    flags_marker.write_text("\n".join(flags), encoding="utf-8")
    return exe


def iters_for(size):
    """给定 LEN, 求出让单 run 恒为 TOTAL_FLOPS_TARGET 的 ITERS。

    必须整除: 否则各尺寸的"每次 run 运算量"就不一致, GFLOPS 横向不可比。
    取 2 的幂做 LEN 时永远整除(ITERS = 2^30 / LEN)。
    """
    denom = 2 * size
    if size < 4 or size % 4 != 0:
        raise BenchError(f"LEN={size} 非法: 需 >=4 且为 4 的倍数(内核按 4-lane ymm 分批)")
    if TOTAL_FLOPS_TARGET % denom != 0:
        raise BenchError(f"LEN={size} 无法整除: {TOTAL_FLOPS_TARGET} / (2*{size}) 不是整数")
    return TOTAL_FLOPS_TARGET // denom


def viper_exe(root):
    return root / "__vpcbuild__" / "output" / "app.exe"


# ------------------------------------------------------------
# 统计与核验
# ------------------------------------------------------------
def mean(vals):
    return sum(vals) / len(vals)


def sample_stddev(vals, m):
    if len(vals) < 2:
        return 0.0
    s = sum((v - m) ** 2 for v in vals)
    return math.sqrt(s / (len(vals) - 1))


def median(vals):
    if not vals:
        return 0.0
    s = sorted(vals)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def geomean(vals):
    """几何均值; 出现 <=0 的样本时退回 0(计时值理论上恒 >0)。"""
    if not vals:
        return 0.0
    if any(v <= 0.0 for v in vals):
        return 0.0
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def pick_stat(name, vals):
    if not vals:
        return 0.0
    if name == "median":
        return median(vals)
    if name == "geomean":
        return geomean(vals)
    return mean(vals)


def trim_group(vals, trim_ratio):
    """按中位数上界裁掉离群点: 保留 v <= median(vals) * trim_ratio。

    返回 (kept, dropped_positions)。trim_ratio <= 0 时等价于不裁剪。
    至少保留中位数本身, 所以 kept 不会为空。
    """
    if trim_ratio <= 0.0 or not vals:
        return list(vals), []
    hi = median(vals) * trim_ratio
    kept = [v for v in vals if v <= hi]
    dropped = [i for i, v in enumerate(vals) if v > hi]
    if not kept:
        return list(vals), []
    return kept, dropped


def analyze_impl(groups_samples, tol, drop_first, trim_ratio, stat_name, metric):
    """groups_samples: list[list[(wall,cpu,flops)]], 每 list = 一个进程内 N 次 run。

    流程: 取主指标 -> 每组按中位数上界裁剪离群点 -> 用 stat_name 统计。
    drop_first=True 时, 首组(索引 0)【仅汇报】: 不计入总体/标准差/最大偏差,
    也不参与差值核验(规避首组效应)。只有一组时自动不排除。
    """
    idx = METRIC_IDX[metric]
    other_idx = 1 - idx
    groups = [[s[idx] for s in g] for g in groups_samples]

    drop_first = drop_first and len(groups) > 1

    trimmed = [trim_group(vals, trim_ratio) for vals in groups]
    stat_groups = trimmed[1:] if drop_first else trimmed

    flat = [v for kept, _ in stat_groups for v in kept]
    overall = pick_stat(stat_name, flat)
    overall_std = sample_stddev(flat, overall)
    max_dev = max((abs(v - overall) for v in flat), default=0.0)
    n_dropped = sum(len(dropped) for _, dropped in stat_groups)

    group_stats = []
    for g, (kept, dropped) in enumerate(trimmed, start=1):
        gm = pick_stat(stat_name, kept)
        group_stats.append({
            "idx": g,
            "raw": groups[g - 1],
            "kept": kept,
            "dropped_idx": dropped,
            "mean": mean(kept),
            "median": median(kept),
            "geomean": geomean(kept),
            "gstat": gm,
            "spread": (max(kept) - min(kept)) if kept else 0.0,
            "dev": gm - overall,
            "in_stat": (not drop_first) or g > 1,   # False 仅组1 被排除时
        })

    adj = [group_stats[k]["gstat"] - group_stats[k - 1]["gstat"]
           for k in range(1, len(group_stats))]

    def rel(x):
        return abs(x) / abs(overall) if overall != 0.0 else (0.0 if x == 0.0 else float("inf"))

    # 差值核验只对计入统计的组判定; 首组(含其相邻差)不作为违规依据
    dev_viol = [gs["idx"] for gs in group_stats if gs["in_stat"] and rel(gs["dev"]) > tol]
    adj_pairs = [(k, k + 1) for k in range(len(adj))
                 if rel(adj[k]) > tol and not (drop_first and k == 0)]
    passed = (not dev_viol) and (not adj_pairs)

    # 另一指标的总体值(同一裁剪与统计口径), 只用于汇总对照
    other_flat = []
    for i, g in enumerate(groups_samples):
        if drop_first and i == 0:
            continue
        kept, _ = trim_group([s[other_idx] for s in g], trim_ratio)
        other_flat.extend(kept)
    overall_other = pick_stat(stat_name, other_flat)

    return {
        "metric": metric, "other_metric": "cpu" if metric == "wall" else "wall",
        "stat": stat_name, "trim_ratio": trim_ratio,
        "overall": overall, "overall_other": overall_other,
        "overall_std": overall_std, "max_dev": max_dev,
        "n": len(flat), "n_dropped": n_dropped,
        "n_stat_groups": len(stat_groups), "drop_first": drop_first,
        "group_stats": group_stats, "adj": adj,
        "passed": passed, "dev_viol": dev_viol, "adj_pairs": adj_pairs,
        "_samples": groups_samples,   # 供 CSV / flops-check
    }


# ------------------------------------------------------------
# 主流程
# ------------------------------------------------------------
def build_arg_parser():
    p = argparse.ArgumentParser(
        description="ViperSpeedBenchmark 编排: 同内核多实现测速 + 组间差值核验"
                    "(每组=单进程内多 run, 线程绑定 + 双计时 + 离群裁剪)")
    p.add_argument("--groups", type=int, default=5, help="组数(默认 5; 每组=一次进程启动)")
    p.add_argument("--runs-per-group", type=int, default=10,
                   help="每组连续 run 次数 = 每次进程启动后打印的 RESULT 行数(默认 10)")
    p.add_argument("--sizes", default=DEFAULT_SIZES,
                   help="逗号分隔的 LEN 列表(多尺寸扫描), 默认 %(default)s。"
                        "每个 LEN 的 ITERS 自动取值, 使单 run 恒为 ≈2.147e9 flops; "
                        "LEN 决定工作集(四份缓冲共 LEN*32 字节), 例如 "
                        "1024=L1 / 16384=L2 / 262144 以上=L3-DRAM")
    p.add_argument("--impl", default=DEFAULT_IMPLS,
                   help="逗号分隔实现子集, 默认 %(default)s。全部 Viper 变体共用同一个 "
                        "app.exe, 只差命令行第 4 个参数 mode: "
                        "viper/viper_scalar=FMA, viper_add/viper_scalar_add=ADD, "
                        "viper_mul/viper_scalar_mul=MUL, viper_sqrt/viper_scalar_sqrt=SQRT "
                        "(前者 SIMD / 后者原生); 另有 c / cpp / py。"
                        "默认已含全部内核变体(plot_bench.py 图1 需要它们的成对数据)")
    p.add_argument("--tol", type=float, default=0.05,
                   help="相对容差, 组对总体与相邻组差值判定阈值(默认 0.05=5%%)")
    p.add_argument("--include-first-group", action="store_true",
                   help="把首组(组1)计入统计与差值核验(默认排除: 组1仅汇报, 规避首组效应)")
    p.add_argument("--metric", default="wall", choices=["wall", "cpu"],
                   help="主指标(默认 wall=QPC 墙钟; cpu=QueryThreadCycleTime 线程 CPU 时间)")
    p.add_argument("--stat", default="mean", choices=["mean", "median", "geomean"],
                   help="组/总体统计量(默认 mean=裁剪后算术平均; 双峰分布建议 median)")
    p.add_argument("--trim-ratio", type=float, default=1.2,
                   help="组内离群裁剪上界 = 组内中位数 × 该系数(默认 1.2); <=0 关闭裁剪")
    p.add_argument("--csv", default=None,
                   help="CSV 输出路径; 省略时自动写成 bench_<时间戳>.csv")
    p.add_argument("--no-csv", action="store_true", help="完全不写 CSV")
    p.add_argument("--cc-dir", default=CC_DIR_DEFAULT, help="clang/clang++ 目录")
    p.add_argument("--report", default="bench_report.txt", help="日志输出文件")
    p.add_argument("--py-mode", default=None, choices=["auto", "numpy", "pure"],
                   help="覆盖 bench_py.py 的运行模式(auto/numpy/pure)")
    p.add_argument("--viper-exe", default=None,
                   help="覆盖 Viper 产物路径(默认 __vpcbuild__/output/app.exe)")
    p.add_argument("--flops-check", action="store_true",
                   help="核对 RESULT 行的 flops 字段是否与 CFG 总运算量一致, "
                        "并报告 flops≈2e9(时长≈1s)的行占比及降规模提示")
    return p


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    args = build_arg_parser().parse_args(argv)

    groups_n = max(1, args.groups)
    runs_n = max(1, args.runs_per_group)
    tol = args.tol
    metric = args.metric
    stat_name = args.stat
    trim_ratio = args.trim_ratio
    drop_first = not args.include_first_group  # 首组效应规避: 组1仅汇报不计入统计
    impl_order = [name for name, _, _ in VIPER_KERNELS] + ["c", "cpp", "py"]
    requested = [x.strip().lower() for x in args.impl.split(",") if x.strip()]
    if requested == ["all"]:
        requested = list(impl_order)
    unknown = [x for x in requested if x not in impl_order]
    if unknown:
        print(f"未知实现名: {unknown} (可选: {','.join(impl_order)})", file=sys.stderr)
        return 2
    impls = [x for x in impl_order if x in requested]

    # ---- 尺寸列表: LEN 决定工作集(四份缓冲共 LEN*32 字节), ITERS 自动算 ----
    sizes = []
    for tok in [t.strip() for t in args.sizes.split(",") if t.strip()]:
        try:
            sizes.append(int(tok))
        except ValueError:
            print(f"--sizes 里有非整数项: {tok!r}", file=sys.stderr)
            return 2
    if not sizes:
        sizes = [DEFAULT_LEN]
    try:
        iters_map = {s: iters_for(s) for s in sizes}
    except BenchError as e:
        print(str(e), file=sys.stderr)
        return 2

    cc_dir = Path(args.cc_dir)
    vpy_exe = Path(args.viper_exe) if args.viper_exe else viper_exe(ROOT)

    env = dict(os.environ)
    if args.py_mode:
        env["VSB_PY_MODE"] = args.py_mode

    lines = []
    def emit(s=""):
        lines.append(s)
        print(s, flush=True)

    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    csv_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    emit("=" * 78)
    emit(" ViperSpeedBenchmark - run_bench.py (线程绑定 + 双计时 + 离群裁剪版)")
    emit(f" 时间       : {stamp}")
    emit(f" 参数       : groups={groups_n}, runs/group={runs_n}, tol={tol:.1%}")
    emit(f" 实现       : {','.join(impls)}")
    emit(f" 负载       : 每次 run ≈ {EXPECT_TOTAL_FLOPS:.3e} flops(LEN×ITERS×2 恒定, 跨尺寸可比)")
    for s in sizes:
        emit(f"              LEN={s:>8}  ITERS={iters_map[s]:>8}   工作集={s * 32 / 1024:>9.1f} KiB")
    emit(" 采集方式   : 每组启动一次子进程, 进程内 2 轮满热身 + 连续计时 N 次")
    emit(" 线程绑定   : 子进程入口 SetThreadAffinityMask 绑到首个可用逻辑核(AFFINITY 行回报)")
    emit(f" 主指标     : {METRIC_LABEL[metric]}   (另一指标同时采集, 汇总里对照)")
    emit(f" 统计量     : {STAT_LABEL[stat_name]}")
    emit(" 离群裁剪   : " + (f"组内保留 <= 中位数×{trim_ratio:g} 的样本" if trim_ratio > 0
                             else "关闭(--trim-ratio<=0)"))
    emit(" 统计口径   : " + ("首组(组1)仅汇报不计入统计(规避首组效应)"
                          if drop_first else "全部组计入统计"))
    emit("=" * 78)

    # ---- 编译 / 检测 ----
    exe_map = {}
    compile_notes = []
    for impl in ("c", "cpp"):
        if impl not in impls:
            continue
        emit(f"[{impl}] 编译 bench_{'c.c' if impl == 'c' else 'cpp.cpp'} -> .native_build/")
        try:
            exe_map[impl] = build_native(cc_dir, impl, ROOT)
            emit(f"[{impl}] 编译成功: {exe_map[impl]}")
        except BenchError as e:
            compile_notes.append((impl, str(e)))
            emit(f"[{impl}] 编译失败: {e}")
            emit(f"[{impl}] 请确认 clang 工具链可用(--cc-dir 可指定目录), 将跳过该实现。")
    viper_wanted = [im for im in impls if im in VIPER_MODE]
    if viper_wanted:
        if vpy_exe.exists():
            for name in viper_wanted:
                exe_map[name] = vpy_exe
            emit(f"[viper] 找到产物: {vpy_exe}")
            emit(f"  {len(viper_wanted)} 个内核变体共用此 exe, 只差命令行 mode 参数: "
                 + ", ".join(f"{n}(mode={VIPER_MODE[n]})" for n in viper_wanted))
        else:
            emit(f"[viper] 未找到 Viper 产物: {vpy_exe}")
            emit("  请先在本工程目录用 tpv 构建 project.vpj(生成 __vpcbuild__/output/app.exe),")
            emit("  本次将跳过全部 Viper 内核变体。")

    # ---- 逐尺寸 × 逐实现跑测(每组=一次进程启动) ----
    # 按"尺寸优先"展开: 同一尺寸下各实现紧挨着跑, 便于横向对比
    results = {}          # key = (size, impl)
    prev_size = None
    for size, impl in [(sz, im) for sz in sizes for im in impls]:
        iters_n = iters_map[size]
        if size != prev_size:
            emit("")
            emit("#" * 78)
            emit(f"# 工作尺寸 LEN={size}  ITERS={iters_n}  四份缓冲共 {size * 32 / 1024:.1f} KiB")
            emit("#" * 78)
            prev_size = size
        # py 始终可运行(用本解释器), 原生/viper 需先成功编译/检出
        if impl == "py":
            pass
        elif impl not in exe_map:
            continue
        emit("")
        emit("-" * 78)
        if impl in VIPER_MODE:
            cmd = [str(exe_map[impl]), str(runs_n), str(size), str(iters_n),
                   str(VIPER_MODE[impl])]
            display = VIPER_DISP[impl] + " 内核, QPC 计时"
        elif impl in ("c", "cpp"):
            cmd = [str(exe_map[impl]), str(runs_n), str(size), str(iters_n)]
            display = ("C 手写 -O2 -mavx2 -mfma(-ffp-contract=fast) 自动向量化单趟 FMA, QPC"
                       if impl == "c" else
                       "C++ 手写(std::vector) -O2 -mavx2 -mfma(-ffp-contract=fast) 单趟 FMA, QPC")
        else:  # py
            cmd = [sys.executable, str(ROOT / "bench_py.py"), str(runs_n), str(size), str(iters_n)]
            display = "Python(numpy 就地 ufunc; 无 numpy 自动纯 Python 降规模回退)"
        emit(f"实现: {impl}    (每组 = 一次进程启动, 进程内满热身 + 连续 {runs_n} 次 run)")
        emit(f" 方式: {display}")
        emit(f" 命令: {' '.join(cmd)}")

        groups = []       # list[list[(wall_us, cpu_us, flops)]]
        meta = {"cfg": None, "check": None, "mode": None, "affinity": None,
                "tsc_hz": None, "kernel_mode": None}
        try:
            for g in range(1, groups_n + 1):
                p = sh_ok(cmd, ROOT, env=env)
                out_text = p.stdout
                if p.returncode != 0:
                    tail = "\n".join(p.stderr.splitlines()[-20:])
                    raise BenchError(
                        f"[{impl}] 组{g} 进程退出码={p.returncode}\n"
                        f"stderr 尾部:\n{tail}\n"
                        f"stdout 前 20 行:\n" + "\n".join(out_text.splitlines()[:20]))
                try:
                    parsed = parse_process_output(out_text, runs_n, impl)
                except ValueError as ve:
                    # 附带 stderr 一起展示, 便于定位崩溃原因
                    stderr_tail = "\n".join(p.stderr.splitlines()[-30:])
                    raise BenchError(
                        f"[{impl}] 组{g} 输出解析失败: {ve}\n"
                        f"stderr 尾部:\n{stderr_tail}") from ve
                groups.append(parsed["samples"])
                if meta["cfg"] is None:
                    meta["cfg"] = parsed["cfg"]
                if meta["check"] is None:
                    meta["check"] = parsed["check"]
                if meta["affinity"] is None:
                    meta["affinity"] = parsed["affinity"]
                if meta["tsc_hz"] is None:
                    meta["tsc_hz"] = parsed["tsc_hz"]
                if meta["kernel_mode"] is None:
                    meta["kernel_mode"] = parsed["kernel_mode"]
                # Viper 内核变体必须真的跑在期望的内核上: 旧版 app.exe 不认命令行第 4
                # 个参数(或内核不全), 会悄悄跑成别的内核 —— 那等于让 SIMD 数据冒充原生,
                # 直接污染 SIMD/原生对照。期望值对不上就显式失败, 不给"看似成功"的机会。
                want_mode = VIPER_MODE.get(impl)
                if want_mode is not None and parsed["kernel_mode"] != want_mode:
                    raise BenchError(
                        f"[{impl}] 产物回报 MODE={parsed['kernel_mode']!r}, 期望 {want_mode}: "
                        "app.exe 可能是旧版(不识别命令行第 4 个参数 mode)或未包含该内核, "
                        "请重新构建 Viper 工程")
                if meta["mode"] is None and parsed["cfg"][1] != iters_n:
                    meta["mode"] = "pure-scale"
                elif meta["mode"] is None:
                    meta["mode"] = "full"
                wall = [s[0] for s in parsed["samples"]]
                cpu = [s[1] for s in parsed["samples"]]
                emit(f"[{impl}] 组{g}/{groups_n} 完成 | 墙钟均值={mean(wall):.1f} us | "
                     f"线程CPU均值={mean(cpu):.1f} us | 墙钟max-min={max(wall) - min(wall):.1f}")
            st = analyze_impl(groups, tol, drop_first, trim_ratio, stat_name, metric)
            st["meta"] = meta
            st["size"] = size
            results[(size, impl)] = st
        except Exception as e:
            # 启动/解析/统计任何异常都要把具体 traceback 打出来, 不再静默跳过
            err_text = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            emit(f"[{impl}] 运行失败(详见 traceback): {e}")
            emit(err_text.rstrip())
            results[(size, impl)] = {"error": str(e), "traceback": err_text}

    # ---- 差值核验 / flops-check 统计输出 ----
    csv_rows = []
    cur_size = None
    for (size, impl), st in results.items():
        if size != cur_size:
            emit("")
            emit("#" * 78)
            emit(f"# 结果汇总: 工作尺寸 LEN={size}  ITERS={iters_map[size]}"
                 f"  四份缓冲共 {size * 32 / 1024:.1f} KiB")
            emit("#" * 78)
            cur_size = size
        if "error" in st:
            emit(f"  {impl}: 运行失败 —— {st['error']}")
            continue
        meta = st.get("meta", {})
        cfg = meta.get("cfg")
        cfg_total = (float(cfg[0]) * float(cfg[1]) * 2.0) if cfg else 0.0
        cfg_txt = f"{cfg[0]}x{cfg[1]}" if cfg else "?"
        emit("")
        emit("=" * 78)
        emit(f"实现 {impl}   CFG(len×iters)={cfg_txt}   总运算量/run={cfg_total:.3e} flops"
             + ("   [纯 Python 降规模]" if meta.get("mode") == "pure-scale" else ""))
        aff = meta.get("affinity")
        emit("线程绑定 AFFINITY  = "
             + (f"CPU#{aff}(SetThreadAffinityMask 成功)"
                if (aff is not None and aff >= 0) else "未绑定/失败(-1)"))
        tsc = meta.get("tsc_hz")
        emit("TSC 标定频率       = "
             + (f"{tsc:.4e} Hz" if tsc else "缺失/0(标定失败: 该实现的周期计数未推进)"))
        km = meta.get("kernel_mode")
        if km is not None:
            emit(f"内核 MODE          = {km}  {VIPER_DISP.get(impl, '(内核名未知)')}")
        all_samples = [s for g in st.get("_samples", []) for s in g]
        n_bad = sum(1 for w, c, _ in all_samples if w > 0.0 and c > w * 1.05)
        if n_bad:
            emit(f"  !! 可疑: {n_bad}/{len(all_samples)} 个样本 cpu_us > wall_us×1.05"
                 " —— 单线程物理上不可能, 请查该实现的 TSC 标定与周期读取")
        emit(f"CHECK(首次) = {meta.get('check', float('nan')):.17g}")
        overall = st["overall"]
        emit(f"主指标 {METRIC_LABEL[st['metric']]}   统计量={STAT_LABEL[st['stat']]}")
        emit(f"总体统计 overall        : {overall:.3f} us  "
             f"(统计 {st['n_stat_groups']}/{groups_n} 组, 有效样本 {st['n']}, "
             f"裁剪剔除 {st['n_dropped']}"
             + ("; 首组仅汇报" if st.get("drop_first") else "") + ")")
        emit(f"对照 {METRIC_LABEL[st['other_metric']]:<24}: {st['overall_other']:.3f} us")
        emit(f"样本标准差 sample std   : {st['overall_std']:.3f} us")
        rel_md = (abs(st["max_dev"]) / abs(overall) if overall else 0.0)
        emit(f"最大偏差 max|raw-overall|: {st['max_dev']:.3f} us  ({rel_md:.3%})")
        if cfg_total:
            emit(f"实测吞吐(按 CFG, 主指标) : {cfg_total / (overall * 1e-6) / 1e9:.3f} GFLOPS")
        emit("-" * 78)
        for gs in st["group_stats"]:
            raw_txt = ", ".join(f"{v:.1f}" for v in gs["raw"])
            dev_rel = abs(gs["dev"]) / abs(overall) if overall else 0.0
            excl = "  [首组: 仅汇报, 不计入统计]" if not gs.get("in_stat", True) else ""
            trim_txt = (f" | 裁剪剔除 {len(gs['dropped_idx'])}/{len(gs['raw'])}"
                        f" 剔除位置={[i + 1 for i in gs['dropped_idx']]}"
                        if gs["dropped_idx"] else " | 裁剪剔除 0")
            emit(f"[组{gs['idx']:>2}] 统计值={gs['gstat']:>11.2f} us | "
                 f"均值={gs['mean']:>11.2f} | 中位数={gs['median']:>11.2f} | "
                 f"几何均值={gs['geomean']:>11.2f}{trim_txt}")
            emit(f"         组内差max-min={gs['spread']:>8.2f} | "
                 f"组-总体={gs['dev']:+10.2f} us ({dev_rel:5.2%}){excl}")
            if gs["idx"] >= 2:
                k = gs["idx"] - 2
                if st.get("drop_first") and k == 0:
                    emit("         相邻组差: n/a(组1 不计入统计)")
                else:
                    adj_val = st["adj"][k]
                    adj_rel = abs(adj_val) / abs(overall) if overall else 0.0
                    emit(f"         相邻组差 组{gs['idx'] - 1}->组{gs['idx']}: {adj_val:+10.2f} us ({adj_rel:5.2%})")
            else:
                emit(f"         相邻组差: n/a(首组)")
            emit(f"         组内 {runs_n} 次原始(主指标): {raw_txt} us")
        emit("-" * 78)
        verdict = "PASS" if st["passed"] else "FAIL"
        emit(f"差值核验(tol={tol:.1%}): {verdict}"
             + ("   [组1 排除: 首组仅汇报]" if st.get("drop_first") else ""))
        if st["passed"]:
            scope = "统计组(排除首组)" if st.get("drop_first") else "全部组"
            emit(f"  {scope}: |组统计值-总体|/总体 <= tol, 且相邻组|差值|/总体 <= tol")
        else:
            if st["dev_viol"]:
                emit(f"  组对总体超差(违规组): {st['dev_viol']}")
            if st["adj_pairs"]:
                emit(f"  相邻组超差(违规对): {['组%d->组%d' % p for p in st['adj_pairs']]}")
        # CSV 原始行(wall/cpu/flops 全出, 附该样本是否被裁剪剔除)
        for gi, gsamples in enumerate(st.get("_samples", []), start=1):
            dropped = set(st["group_stats"][gi - 1]["dropped_idx"])
            for ri, (wv, cv, fv) in enumerate(gsamples, start=1):
                csv_rows.append([impl, cfg[0] if cfg else "", cfg[1] if cfg else "",
                                 gi, ri, wv, cv, fv, meta.get("check", ""),
                                 1 if (ri - 1) in dropped else 0])

    # ---- flops-check ----
    if args.flops_check:
        emit("")
        emit("=" * 78)
        emit("flops-check 核验(需 --flops-check): 校验 CFG 总运算量≈2.147e9 及")
        emit("RESULT 行 flops 字段与 us 的换算一致性(flops = CFG_total / 墙钟秒)")
        emit("-" * 78)
        any_issue = False
        for (size, impl), st in results.items():
            if "error" in st:
                continue
            cfg = st.get("meta", {}).get("cfg")
            if not cfg:
                emit(f"  [LEN={size} {impl}] 无 CFG, 无法核验")
                any_issue = True
                continue
            cfg_total = float(cfg[0]) * float(cfg[1]) * 2.0
            near = abs(cfg_total - EXPECT_TOTAL_FLOPS) / EXPECT_TOTAL_FLOPS <= 0.05
            bad = []
            ok_n = 0
            near_2e9_n = 0
            flat = [(w, c, f) for g in st.get("_samples", []) for (w, c, f) in g]
            for row_w, row_c, row_f in flat:
                if row_w <= 0:
                    bad.append((row_w, row_f))
                    continue
                expect_f = cfg_total / (row_w * 1e-6)
                if abs(row_f - expect_f) / expect_f > 0.01:
                    bad.append((row_w, row_f))
                else:
                    ok_n += 1
                if abs(row_f - EXPECT_TOTAL_FLOPS) / EXPECT_TOTAL_FLOPS <= 0.5:
                    near_2e9_n += 1
            total_n = len(flat)
            emit(f"  [LEN={size} {impl}] CFG={cfg[0]}x{cfg[1]}  单 run 总运算量={cfg_total:.3e} flops"
                 + ("  (≈2.147e9 ✓)" if near else "  (≠2e9: 降规模配置, 见 README §8)"))
            emit(f"      RESULT 行 flops 字段与墙钟 us 换算一致: {ok_n}/{total_n}"
                 + ("" if not bad else f"  <<< 不一致示例 {bad[:3]}"))
            emit(f"      flops≈2e9 的行(即单 run 时长≈1.0s): {near_2e9_n}/{total_n} "
                 "(说明: RESULT 的 flops=总运算量/秒, 只有时长恰≈1s 才≈2e9)")
            if not near:
                any_issue = True
            if bad:
                any_issue = True
        if not any_issue:
            emit("  全部实现 CFG 与 flops 换算均正常。")

    # ---- 汇总 ----
    emit("")
    emit("=" * 78)
    emit(f"汇总: 各实现总体统计值(us) 与 实测 GFLOPS(按各自 CFG); 主指标={METRIC_LABEL[metric]}")
    cur_size = None
    for (size, impl), st in results.items():
        if size != cur_size:
            emit("")
            emit(f"  ==== LEN={size}  (工作集 {size * 32 / 1024:.1f} KiB) ====")
            cur_size = size
        if "error" in st:
            emit(f"  {impl:<6}: 失败(见上方 traceback)")
            continue
        cfg = st.get("meta", {}).get("cfg")
        cfg_total = (float(cfg[0]) * float(cfg[1]) * 2.0) if cfg else 0.0

        def gfl(us):
            return cfg_total / (us * 1e-6) / 1e9 if (cfg_total and us) else float("nan")

        main_us = st["overall"]
        other_us = st["overall_other"]
        lab_main = "墙钟" if st["metric"] == "wall" else "CPU "
        lab_other = "CPU " if st["metric"] == "wall" else "墙钟"
        aff = st.get("meta", {}).get("affinity")
        aff_txt = f"CPU#{aff}" if (aff is not None and aff >= 0) else "未绑定"
        emit(f"  {impl:<6}: [{lab_main}] {main_us:>11.3f} us {gfl(main_us):>8.3f} GFLOPS"
             f" | [{lab_other}] {other_us:>11.3f} us {gfl(other_us):>8.3f} GFLOPS"
             f" | {aff_txt} | {'PASS' if st['passed'] else 'FAIL'}")
    if not any(im == "viper" for (_, im) in results):
        emit("  (viper 已跳过: 请先构建 __vpcbuild__/output/app.exe)")
    for impl, note in compile_notes:
        emit(f"  ({impl} 已跳过: 编译失败, 见上方)")
    emit("=" * 78)

    report = ROOT / args.report
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    emit(f"\n日志已写入: {report}")

    # CSV 默认自动导出(带时间戳), 便于事后画图; --no-csv 可关闭
    if not args.no_csv:
        csv_path = ROOT / (args.csv if args.csv else f"bench_{csv_stamp}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["impl", "len", "iters", "group", "run",
                        "wall_us", "cpu_us", "flops", "check", "trimmed"])
            w.writerows(csv_rows)
        emit(f"原始数据已写入: {csv_path}")
        emit(f"  (出图: python plot_bench.py {csv_path.name})")

    if not results or all("error" in st for st in results.values()):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
