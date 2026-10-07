# -*- coding: utf-8 -*-
"""
bench_py.py - Python 基准实现 (进程内多次 run, 单进程输出多行)

内核(与 App/main.vp / bench_c.c / bench_cpp.cpp 一致):
    out[i] = a[i] * b[i] + c[i]      LEN = 16384, 每次 run = ITERS = 65536 趟
每次 run 总 flops = LEN*ITERS*2 ≈ 2.147e9, 与其它实现同规模可比。

用法: python bench_py.py [runs]      # runs 默认 1; 0 = 仅热身

模式(由环境变量 VSB_PY_MODE 控制, 默认 auto):
  * auto / numpy: 需要 numpy。内核用预分配 out 缓冲 + 两个就地 ufunc:
        numpy.multiply(na, nb, out=out);  numpy.add(out, nc, out=out)
    逐元素两次舍入(a*b 先舍入一次, +c 再舍入一次)。注意 C/C++ 与 Viper 现在都走
    单趟 FMA(单次舍入, 编译加 -ffp-contract=fast), 所以 numpy 的逐元素结果与它们
    可能差 1 ulp —— 这是舍入口径差异, 不是错误。
  * auto 无 numpy / pure: 退化为纯 Python 循环。纯 Python 无法在合理时间内完成
    65536 趟(LEN=16384), 因此自动把每 run 趟数降为 PY_PURE_ITERS=8 并如实打印
    CFG <len> <iters>; run_bench 的 --flops-check 会据此提示"非 2e9 配置"。
    每元素仍两次舍入(CPython 不把 a*b+c 收缩成 FMA), 与 numpy 逐位一致。

计时: time.perf_counter_ns (Windows 底层即 QPC) -> 微秒, 只包住 ITERS 趟内核。
      线程 CPU 时间用 ctypes 调 kernel32.QueryThreadCycleTime 取本线程消耗的 CPU
      周期数, 再用入口标定出的 TSC 频率换算成微秒; 两者同口径包住同一段内核。
      (不用 GetThreadTimes: 它的记账粒度跟随调度时钟节拍, 实测硬地板 15.625ms,
       几十毫秒的 run 误差 ±40%, 且在 Win10 2004+ 无法用 timeBeginPeriod 改善。)
线程绑定: 入口用 ctypes 调 kernel32.SetThreadAffinityMask 绑到首个可用逻辑核
      (与 C/C++/Viper 实现绑同一个核), 以 AFFINITY <core> 行回报, -1 = 失败。
热身: 进程内先做 2 轮满负荷热身(每轮 = 实际趟数 + 升序读回 out 求校验和)。
输出(进程开头一次 CFG + 一次 AFFINITY, 之后每次 run 一行):
    CFG      <len> <iters>
    AFFINITY <core>
    RESULT   <wall_us> <cpu_us> <flops>   # flops = (LEN*ITERS*2)/墙钟秒
    CHECK    <sum>                        # 最后一次 run 留下的 out 升序累加
"""
import ctypes
import math
import os
import sys
import time
from ctypes import wintypes

# 负载约束: LEN × ITERS × 2 恒为 ≈2.147e9 flops/run(与其它实现可比),
# 只调 LEN 改"工作集": LEN=16384 -> 四份共 512 KiB, 稳居本机每核 1 MiB 私有 L2。
LEN = 16384           # 元素数(与 C/C++/Viper 一致)
ITERS = 65536         # 每 run 的内核趟数(numpy/自动模式)
PY_PURE_ITERS = 8     # 纯 Python 回退趟数(降规模, 保证可测且如实报 CFG)
RUNS_MAX = 200

_LCG_A = 48271.0
_LCG_M = 2147483647.0
_LCG_SEED = 12345.0

# win32 SetThreadPriority 的 THREAD_PRIORITY_HIGHEST
_THREAD_PRIORITY_HIGHEST = 2

# ------------------------------------------------------------
# Win32 线程绑定 / 线程 CPU 时间(经 ctypes), 与 C/C++/Viper 实现同口径
# ------------------------------------------------------------
_IS_WINDOWS = os.name == "nt"

if _IS_WINDOWS:
    _K32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _K32.GetCurrentThread.restype = wintypes.HANDLE
    _K32.SetThreadAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
    _K32.SetThreadAffinityMask.restype = ctypes.c_size_t
    _K32.SetThreadPriority.argtypes = [wintypes.HANDLE, ctypes.c_int]
    _K32.SetThreadPriority.restype = wintypes.BOOL
    _K32.QueryThreadCycleTime.argtypes = [wintypes.HANDLE,
                                          ctypes.POINTER(ctypes.c_ulonglong)]
    _K32.QueryThreadCycleTime.restype = wintypes.BOOL
    _K32.QueryPerformanceCounter.argtypes = [ctypes.POINTER(ctypes.c_longlong)]
    _K32.QueryPerformanceCounter.restype = wintypes.BOOL
    _K32.QueryPerformanceFrequency.argtypes = [ctypes.POINTER(ctypes.c_longlong)]
    _K32.QueryPerformanceFrequency.restype = wintypes.BOOL
else:
    _K32 = None


def _bind_first_available_core():
    """把当前线程绑到首个可用逻辑核, 返回 CPU 序号; 失败/非 Windows 返回 -1。

    从 1<<0 逐个试, 第一个被系统接受的位即亲和掩码里编号最小的核 ——
    四个实现会绑到同一个核, 横向对比才有意义。
    """
    if _K32 is None:
        return -1
    h = _K32.GetCurrentThread()
    for i in range(64):
        if _K32.SetThreadAffinityMask(h, ctypes.c_size_t(1 << i)):
            return i
    return -1


def _thread_cycles():
    """本线程累计消耗的 CPU 周期数(QueryThreadCycleTime; 非 Windows 返回 0)。

    现代 x86 的 TSC 是 invariant 的, 与核心实际频率无关, 所以"周期数"就是
    "本线程真正在 CPU 上跑过的时间", 对抢占与降频都不敏感。
    """
    if _K32 is None:
        return 0
    c = ctypes.c_ulonglong(0)
    _K32.QueryThreadCycleTime(_K32.GetCurrentThread(), ctypes.byref(c))
    return c.value


def _calibrate_tsc_hz():
    """用 QPC 标定 TSC 频率(周期/秒): 忙等 30ms 数周期, 重复 3 次取最大。

    为什么取最大: 标定窗口内本线程一旦被抢占, 消耗周期就少于墙钟流逝时间,
    算出 hz 偏小 -> 后续 cpu_us 系统性偏大(实测约 +0.13%, 表现为"cpu_us 偶尔
    比 wall_us 还大"这种看着像 bug 的伪影)。取三次最大值 = 挑最没被打断的一次。
    必须在 _bind_first_available_core() 之后调用, 否则跨核的 TSC 可能不同步。
    返回 0.0 表示标定失败。
    """
    if _K32 is None:
        return 0.0
    qf = ctypes.c_longlong(0)
    _K32.QueryPerformanceFrequency(ctypes.byref(qf))
    target = qf.value * 30 // 1000          # 30ms 对应的 QPC 计次
    best = 0.0
    for _ in range(3):
        a = ctypes.c_longlong(0)
        b = ctypes.c_longlong(0)
        c0 = _thread_cycles()
        _K32.QueryPerformanceCounter(ctypes.byref(a))
        while True:
            _K32.QueryPerformanceCounter(ctypes.byref(b))
            if b.value - a.value >= target:
                break
        c1 = _thread_cycles()
        dt = b.value - a.value
        if dt <= 0 or c1 <= c0:
            continue
        hz = (c1 - c0) * qf.value / dt
        if hz > best:
            best = hz
    return best


def _fill_abc(n):
    """逐元素填充 a/b/c: 依次消耗同一个 LCG 流(a[i], b[i], c[i])。

    返回三个 float 列表, 与 C(fmod)/Viper(vipermath.fmod) 逐位一致。
    """
    a = [0.0] * n
    b = [0.0] * n
    c = [0.0] * n
    state = _LCG_SEED
    for i in range(n):
        state = math.fmod(state * _LCG_A, _LCG_M)
        a[i] = state / _LCG_M
        state = math.fmod(state * _LCG_A, _LCG_M)
        b[i] = state / _LCG_M
        state = math.fmod(state * _LCG_A, _LCG_M)
        c[i] = state / _LCG_M
    return a, b, c


def _checksum_seq(values):
    """升序逐元素累加(纯 Python double 左折叠, 与 C/Viper 校验顺序一致)"""
    s = 0.0
    for v in values:
        s += v
    return s


def _run_numpy_pass(np_, na, nb, nc, out):
    """单趟内核: out = a*b + c (两趟就地 ufunc, 逐元素两次舍入)"""
    np_.multiply(na, nb, out=out)
    np_.add(out, nc, out=out)


def _argv_int(idx, default):
    """取第 idx 个命令行整数参数(1 起, 即 sys.argv[idx]); 缺失/非法返回 default。"""
    if len(sys.argv) > idx:
        try:
            return int(sys.argv[idx])
        except ValueError:
            pass
    return default


def main():
    # 命令行: bench_py.py [runs] [len] [iters]
    #   省略则用本文件顶部的默认值; run_bench.py 多尺寸扫描时会显式传入。
    runs = max(0, min(_argv_int(1, 1), RUNS_MAX))
    n = _argv_int(2, LEN)
    iters_req = _argv_int(3, ITERS)
    if n < 4 or n % 4 != 0 or iters_req < 1:
        sys.stderr.write(
            f"bench_py: bad args len={n}(需 >=4 且为 4 的倍数) iters={iters_req}\n")
        return 2

    # 线程绑定(必做) + 提优先级: 与 C/C++/Viper 同口径, 并压低背景进程抢占的尖峰
    core = _bind_first_available_core()
    _K32.SetThreadPriority(_K32.GetCurrentThread(), _THREAD_PRIORITY_HIGHEST)
    # TSC 频率标定(必须在绑核之后): 把 QueryThreadCycleTime 的周期数换算成微秒
    tsc_hz = _calibrate_tsc_hz()

    def cpu_us_from(cy0, cy1):
        """周期差 -> 微秒; 标定失败(tsc_hz==0)时给 0, 表现为一眼可见的异常。"""
        return (cy1 - cy0) * 1000000.0 / tsc_hz if tsc_hz > 0.0 else 0.0

    mode = os.environ.get("VSB_PY_MODE", "auto").strip().lower()
    numpy = None
    if mode in ("auto", "numpy"):
        try:
            import numpy as _np
            numpy = _np
        except Exception:
            numpy = None
        if mode == "numpy" and numpy is None:
            sys.stderr.write("bench_py: 强制 numpy 模式但本机无 numpy; "
                             "请改 VSB_PY_MODE=auto/pure 或安装 numpy\n")
            return 1

    use_numpy = numpy is not None

    # 纯 Python 回退把趟数降规模(否则跑不完); n 仍用请求值, 如实打印 CFG
    iters = iters_req if use_numpy else PY_PURE_ITERS

    print(f"CFG {n} {iters}", flush=True)
    print(f"AFFINITY {core}", flush=True)
    print(f"TSC {tsc_hz:.4e}", flush=True)

    a, b, c = _fill_abc(n)

    if use_numpy:
        # 值逐位不变: list -> float64 ndarray(仅一次转换)
        na = numpy.array(a, dtype=numpy.float64)
        nb = numpy.array(b, dtype=numpy.float64)
        nc = numpy.array(c, dtype=numpy.float64)
        out = numpy.empty(n, dtype=numpy.float64)

        def warm_round():
            for _ in range(iters):
                _run_numpy_pass(numpy, na, nb, nc, out)

        def timed_round():
            cy0 = _thread_cycles()
            t0 = time.perf_counter_ns()
            for _ in range(iters):
                _run_numpy_pass(numpy, na, nb, nc, out)
            t1 = time.perf_counter_ns()
            cy1 = _thread_cycles()
            return (t1 - t0) * 1e-3, cpu_us_from(cy0, cy1)

        def readback_sum():
            return _checksum_seq(out.tolist())
    else:
        # 纯 Python 回退: 列表就地写 out
        la, lb, lc = a, b, c
        out = [0.0] * n

        def warm_round():
            for _ in range(iters):
                for i in range(n):
                    out[i] = la[i] * lb[i] + lc[i]

        def timed_round():
            cy0 = _thread_cycles()
            t0 = time.perf_counter_ns()
            for _ in range(iters):
                for i in range(n):
                    out[i] = la[i] * lb[i] + lc[i]
            t1 = time.perf_counter_ns()
            cy1 = _thread_cycles()
            return (t1 - t0) * 1e-3, cpu_us_from(cy0, cy1)

        def readback_sum():
            return _checksum_seq(out)

    if not use_numpy:
        sys.stderr.write("bench_py: numpy 不可用, 使用纯 Python 回退"
                         f"(LEN={n}, 趟数={iters}, 总 flops 非 2e9)\n")

    total_flops = float(n) * float(iters) * 2.0

    # 2 轮满负荷热身(每轮 = 实际趟数 + 读回 out 求校验和)
    last_chk = 0.0
    for _ in range(2):
        warm_round()
        last_chk = readback_sum()

    # 连续计时 runs 次; 计时只包住 iters 趟内核, 墙钟与线程 CPU 同时包住
    for _ in range(runs):
        wall_us, cpu_us = timed_round()
        flops = total_flops / (wall_us * 1e-6)
        print(f"RESULT {wall_us:.3f} {cpu_us:.3f} {flops:.0f}", flush=True)
        last_chk = readback_sum()

    print(f"CHECK {last_chk:.17g}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
