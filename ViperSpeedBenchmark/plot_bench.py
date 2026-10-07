# -*- coding: utf-8 -*-
"""
plot_bench.py - 读 run_bench.py 导出的 CSV, 生成 4 张对比图(无需重跑基准)

用法:
    python plot_bench.py                          # 自动取最新的 bench_*.csv
    python plot_bench.py bench_20260910_2031.csv
    python plot_bench.py bench_xxx.csv --metric cpu
    python plot_bench.py bench_xxx.csv --out-dir figs --prefix run1

产出(4 张 PNG, 默认与 CSV 同目录, 前缀 = CSV 文件名):
    <前缀>_1_simd_vs_scalar.png   左: ADD/MUL/SQRT 的 原生 vs SIMD 延迟对比(柱状)
                                  右: 各操作"比原生快多少倍"(柱状 + xN.NN 标注)
    <前缀>_2_exec_time.png        各实现 Execution Time 折线(叠加每次 run 的实测点)
    <前缀>_3_overall_mean.png     各实现 Overall Mean Execution Time 对比
    <前缀>_4_gflops.png           各实现 Measured GFLOPS 吞吐对比

统计口径与 run_bench.py 报告一致: CSV 里 trimmed=1 的样本(被离群裁剪剔除)不计入。
图1 需要 CSV 里同时有 ADD/MUL/SQRT 的 原生+SIMD 成对数据, 例如:
    python run_bench.py --impl viper_add,viper_scalar_add,viper_mul,viper_scalar_mul,viper_sqrt,viper_scalar_sqrt
缺哪一对就少画哪一根, 全缺则整张图跳过并提示。
"""
import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # 无 GUI 后端, 只存文件
import matplotlib.pyplot as plt             # noqa: E402
import numpy as np                          # noqa: E402

# 图表标题/坐标轴都是中文: 必须显式指定带 CJK 字形的字体, 否则全是方块
matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Microsoft JhengHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent

# 实现名 -> (显示名, 颜色), 固定顺序便于多次运行配色一致
# 显示名刻意短: 图内文字统一英文, 图例太长会把绘图区挤扁
IMPL_STYLE = [
    ("viper", "Viper SIMD", "#d62728"),
    ("viper_scalar", "Viper Scalar", "#ff7f0e"),
    ("c", "C (-O2 -mavx2 -mfma)", "#1f77b4"),
    ("cpp", "C++ (std::vector)", "#2ca02c"),
    ("py", "Python + numpy", "#9467bd"),
]

# 图1: 三种操作 -> (显示名, SIMD 实现名, 原生实现名)。
# 每一对都是同一个 app.exe 的两个 mode, 除"是否向量化"外无任何差异
# (见 App/main.vp 的 KERNEL_* 与 run_bench.py 的 VIPER_KERNELS)。
KERNEL_PAIRS = [
    ("ADD",  "viper_add",  "viper_scalar_add"),
    ("MUL",  "viper_mul",  "viper_scalar_mul"),
    ("SQRT", "viper_sqrt", "viper_scalar_sqrt"),
]

# 图1 配色: 原生(标量) vs SIMD
COL_NATIVE = "#8c8c8c"
COL_SIMD = "#d62728"


def style_for(impl):
    """实现名 -> (显示名, 颜色); 未登记的实现给灰色, 不静默改名。"""
    for name, disp, color in IMPL_STYLE:
        if name == impl:
            return disp, color
    return impl, "#7f7f7f"


def metric_txt(metric):
    """耗时口径 -> 英文短名(图内文字一律用英文, 避免中文长句撑爆标题)。"""
    return "wall clock" if metric == "wall" else "thread CPU"


def human_kib(size):
    """LEN(元素数, double) -> 工作集描述(四份缓冲共 LEN*32 字节)"""
    kib = size * 32 / 1024.0
    if kib >= 1024:
        return f"{size}\n({kib / 1024:.0f} MiB)"
    return f"{size}\n({kib:.0f} KiB)"


def load_csv(path):
    """读 CSV -> {(impl, len): {"wall":[...], "cpu":[...], "iters":n, "trimmed":n,
                               "groups": {组号: {"wall":[...], "cpu":[...]}}}}

    注意 iters 要留着: 每个尺寸的 LEN×ITERS×2 = 该尺寸下单 run 的总运算量,
    是 GFLOPS 换算的唯一依据(py 纯解释器回退会降规模, iters 与其它实现不同)。
    groups 供"只有一个 LEN 时改用组号当横轴"用(见 fig_exec_time)。
    """
    data = {}
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            key = (row["impl"], int(row["len"]))
            rec = data.setdefault(key, {"wall": [], "cpu": [], "iters": int(row["iters"]),
                                        "trimmed": 0, "groups": {}})
            if int(row["trimmed"]) == 1:
                rec["trimmed"] += 1
                continue
            wv = float(row["wall_us"])
            cv = float(row["cpu_us"])
            rec["wall"].append(wv)
            rec["cpu"].append(cv)
            grp = rec["groups"].setdefault(int(row["group"]), {"wall": [], "cpu": []})
            grp["wall"].append(wv)
            grp["cpu"].append(cv)
            if int(row["iters"]) != rec["iters"]:
                rec["iters"] = int(row["iters"])   # 防御: 同名 key 的 iters 应一致
    return data


def series(rec, metric):
    """取某实现某尺寸在指定口径下的有效样本(计时值必须 > 0)。"""
    v = np.asarray(rec[metric], dtype=float)
    return v[np.isfinite(v) & (v > 0.0)]


def annotate_bars(ax, xs, ys, spec=".1f", dy_ratio=0.02, fontsize=9, color="black"):
    """在柱顶标数值(xs/ys 为等长序列)。

    spec 用 format() 的格式说明符(如 ".1f" / ",.0f"), 不用 % 风格 ——
    Python 的 % 格式化不支持千分位(%,.0f 会直接抛 ValueError)。
    """
    span = (max(ys) - min(ys)) if len(ys) > 1 else (abs(ys[0]) if ys else 1.0)
    dy = (span if span > 0 else 1.0) * dy_ratio
    for x, y in zip(xs, ys):
        ax.text(x, y + dy, format(y, spec), ha="center", va="bottom",
                fontsize=fontsize, color=color)


# ------------------------------------------------------------
# 图1: ViperSIMD - ADD / MUL / SQRT 的 原生 vs SIMD(左右双面板)
# ------------------------------------------------------------
def fig_simd_vs_scalar(data, sizes, args, metric, out_path):
    """左面板 = 每种操作(ADD/MUL/SQRT)的 原生 vs SIMD 延迟柱状对比;
    右面板 = 对应的 SIMD 加速比(原生延迟 / SIMD 延迟)。

    只取一个 LEN(CSV 里最小的那个): 加速比是"指令级/微架构"特性, 在能驻留私有
    L2 的小尺寸上测得最干净; 想看跨尺寸的访存行为请用图2。
    """
    if not sizes:
        print("[图1] 跳过: CSV 里没有样本")
        return False
    use_size = sizes[0]
    rows = []      # (op, native_us, simd_us)
    for op, simd_impl, native_impl in KERNEL_PAIRS:
        rs = data.get((simd_impl, use_size))
        rn = data.get((native_impl, use_size))
        if not rs or not rn:
            continue
        vs = series(rs, metric)
        vn = series(rn, metric)
        if vs.size == 0 or vn.size == 0:
            continue
        rows.append((op, float(vn.mean()), float(vs.mean())))
    if not rows:
        need = ",".join(f"{a},{b}" for _, a, b in KERNEL_PAIRS)
        print("[图1] 跳过: CSV 里没有 ADD/MUL/SQRT 的 原生+SIMD 成对数据")
        print(f"  该 CSV 实际包含: {sorted({k[0] for k in data})}")
        print("  直接重跑一次即可(run_bench.py 默认已含这些内核变体): python run_bench.py")
        print(f"  或只补跑内核变体: --impl {need}")
        return False

    ops = [r[0] for r in rows]
    native_us = [r[1] for r in rows]
    simd_us = [r[2] for r in rows]
    speedup = [(n / s if s > 0 else float("nan")) for n, s in zip(native_us, simd_us)]
    best = max(speedup)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 6.4))
    x = np.arange(len(ops), dtype=float)
    w = 0.36

    # ---- 左: 延迟对比(每组内 左=原生 / 右=SIMD) ----
    axL.bar(x - w / 2, native_us, w, color=COL_NATIVE, alpha=0.9,
            label="Native (scalar)")
    axL.bar(x + w / 2, simd_us, w, color=COL_SIMD, alpha=0.9,
            label="ViperSIMD (AVX2 x4)")
    axL.set_xticks(x)
    axL.set_xticklabels(ops, fontsize=11)
    axL.set_xlabel("Operation", fontsize=10)
    axL.set_ylabel(f"Latency per Run  (us, {metric_txt(metric)})", fontsize=10)
    axL.set_title(f"Latency: Native vs ViperSIMD\nLEN={use_size}", fontsize=12)
    axL.grid(alpha=0.3, axis="y")
    axL.legend(fontsize=10)
    hi = max(native_us + simd_us)
    for xi, y in zip(x - w / 2, native_us):
        axL.text(xi, y + hi * 0.01, f"{y:,.0f}", ha="center", va="bottom", fontsize=9)
    for xi, y in zip(x + w / 2, simd_us):
        axL.text(xi, y + hi * 0.01, f"{y:,.0f}", ha="center", va="bottom", fontsize=9)

    # ---- 右: 比原生快多少倍 ----
    axR.bar(x, speedup, 0.5, color=COL_SIMD, alpha=0.9)
    axR.axhline(1.0, color="#555555", lw=1.2, ls="--")
    axR.text(len(ops) - 0.55, 1.0, "x1 = no gain", va="bottom", ha="right",
             fontsize=9, color="#555555")
    axR.set_xticks(x)
    axR.set_xticklabels(ops, fontsize=11)
    axR.set_xlabel("Operation", fontsize=10)
    axR.set_ylabel("Speed-up  (native latency / ViperSIMD latency)", fontsize=10)
    axR.set_title("SIMD Speed-up over Native", fontsize=12)
    axR.grid(alpha=0.3, axis="y")
    axR.set_ylim(0, best * 1.30 if best > 0 else 2.0)
    for xi, sp in zip(x, speedup):
        axR.text(xi, sp + best * 0.02, f"x{sp:.2f}", ha="center", va="bottom",
                 fontsize=13, fontweight="bold")

    fig.suptitle(f"{args.csv_name}\n"
                 f"bar = mean of all runs  ·  metric = {metric_txt(metric)}"
                 f"  ·  trimmed outliers excluded"
                 f"  ·  latency: lower is better", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)
    print(f"[图1] Native vs SIMD 已保存: {out_path}")
    return True


# ------------------------------------------------------------
# 图2: 各实现 Execution Time 折线 + 每次 run 的实测点
# ------------------------------------------------------------
def fig_exec_time(data, sizes, impls, args, metric, out_path):
    # 只有一个 LEN 时横轴没东西可展开(折线会退化成孤点), 改用"组号"当横轴:
    # 每组 = 一次进程内多次 run, 于是单尺寸下依然能看到折线与组间离散。
    single = len(sizes) == 1
    if single:
        xkeys = sorted({g for rec in data.values() for g in rec["groups"]})
        xlabel = f"Group Index   (LEN={sizes[0]}; one group = one process run)"
        xticklabels = [str(g) for g in xkeys]
        note = ("line = group mean, dots = individual runs\n"
                f"single LEN={sizes[0]}: X axis = group index")
    else:
        xkeys = list(sizes)
        xlabel = "Working Set  a/b/c/out  (LEN)"
        xticklabels = [human_kib(s) for s in sizes]
        note = "line = per-size mean, dots = individual runs"

    def samples_at(rec, xi):
        """第 xi 个横轴位置上的有效样本(计时值必须 > 0)。"""
        raw = rec["groups"].get(xkeys[xi]) if single else rec
        if not raw:
            return np.array([])
        v = np.asarray(raw[metric], dtype=float)
        return v[np.isfinite(v) & (v > 0.0)]

    def rec_of(impl, xi):
        return data.get((impl, sizes[0])) if single else data.get((impl, xkeys[xi]))

    fig, ax = plt.subplots(figsize=(12.5, 6.8))
    x = np.arange(len(xkeys), dtype=float)
    n = max(1, len(impls))
    allv = []

    for k, impl in enumerate(impls):
        disp, color = style_for(impl)
        off = (k - (n - 1) / 2.0) * 0.11        # 各实现的散点横向错开, 避免互相压盖
        xs, ys = [], []
        for i in range(len(xkeys)):
            rec = rec_of(impl, i)
            if not rec:
                continue
            v = samples_at(rec, i)
            if v.size == 0:
                continue
            xs.append(x[i])
            ys.append(float(v.mean()))
            ax.scatter(np.full(v.shape, x[i] + off), v, s=16, color=color,
                       alpha=0.35, linewidths=0, zorder=2)
            allv.extend(v.tolist())
        if xs:
            ax.plot(xs, ys, marker="o", ms=8, lw=2.2, color=color,
                    label=f"{disp}  (mean)", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(f"Execution Time per Run  (us, {metric_txt(metric)})", fontsize=10)
    ax.set_title("Execution Time Comparison\n" + note, fontsize=12)

    # 动态范围很大(如 numpy 与 SIMD 差百倍)时自动取对数, 否则小值被压成一条线
    use_log = args.y_scale == "log"
    if args.y_scale == "auto" and allv:
        lo, hi = min(allv), max(allv)
        use_log = lo > 0 and (hi / lo) > 30.0
    if use_log:
        ax.set_yscale("log")
    ax.set_axisbelow(True)
    ax.grid(alpha=0.3, which="both")

    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=9.5, ncol=2)
    fig.suptitle(f"{args.csv_name}  ·  y-axis: {'log' if use_log else 'linear'}"
                 f"  ·  trimmed outliers excluded"
                 f"  ·  implementations offset horizontally", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)
    print(f"[图2] Execution Time 折线已保存: {out_path}")
    return True


# ------------------------------------------------------------
# 图3: Overall Mean Execution Time 对比
# ------------------------------------------------------------
def fig_overall_mean(data, sizes, impls, args, metric, out_path):
    labels, means, stds, colors = [], [], [], []
    for impl in impls:
        vals = [series(data[(impl, s)], metric) for s in sizes if (impl, s) in data]
        vals = [v for v in vals if v.size]
        if not vals:
            continue
        v = np.concatenate(vals)
        disp, color = style_for(impl)
        labels.append(disp)
        means.append(float(v.mean()))
        stds.append(float(v.std(ddof=1)) if v.size > 1 else 0.0)
        colors.append(color)
    if not labels:
        print("[图3] 跳过: 没有有效样本")
        return False

    fig, ax = plt.subplots(figsize=(11.0, 6.4))
    xs = np.arange(len(labels), dtype=float)
    ax.bar(xs, means, 0.6, color=colors, alpha=0.9,
           yerr=stds, capsize=6, ecolor="#333333")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel(f"Mean Execution Time per Run  (us, {metric_txt(metric)})", fontsize=10)
    ax.set_title("Overall Mean Execution Time\n"
                 "all valid samples pooled  ·  error bar = std dev"
                 "  ·  in-bar xN.NN = relative to fastest", fontsize=12)
    ax.grid(alpha=0.3, axis="y")
    annotate_bars(ax, xs, means, spec=",.0f" if max(means) > 10000 else ".1f")

    # 柱身只标"相对最快者的倍数"(越小越快); 最快的那根直接标"最快"
    fastest = min(means)
    for x, y in zip(xs, means):
        mult = y / fastest
        ax.text(x, y * 0.55, "fastest" if mult <= 1.0001 else f"x{mult:.2f}",
                ha="center", va="center", fontsize=9.5, color="white", fontweight="bold")
    fig.suptitle(f"{args.csv_name}\n"
                 f"per-run work is constant (LEN×ITERS×2),"
                 f" so absolute times are comparable", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)
    print(f"[图3] Overall Mean Execution Time 已保存: {out_path}")
    return True


# ------------------------------------------------------------
# 图4: Measured GFLOPS 吞吐对比
# ------------------------------------------------------------
def fig_gflops(data, sizes, impls, args, metric, out_path):
    labels, means, stds, colors = [], [], [], []
    for impl in impls:
        per_run = []
        for s in sizes:
            rec = data.get((impl, s))
            if not rec:
                continue
            v = series(rec, metric)
            if v.size == 0:
                continue
            # 该尺寸单 run 的总运算量 = LEN × ITERS × 2 (读 CSV 的 iters, 兼容降规模)
            flops_run = float(s) * float(rec["iters"]) * 2.0
            per_run.append(flops_run / (v * 1e-6) / 1e9)     # 每次 run 的 GFLOPS
        if not per_run:
            continue
        v = np.concatenate(per_run)
        disp, color = style_for(impl)
        labels.append(disp)
        means.append(float(v.mean()))
        stds.append(float(v.std(ddof=1)) if v.size > 1 else 0.0)
        colors.append(color)
    if not labels:
        print("[图4] 跳过: 没有有效样本")
        return False

    fig, ax = plt.subplots(figsize=(11.0, 6.4))
    xs = np.arange(len(labels), dtype=float)
    ax.bar(xs, means, 0.6, color=colors, alpha=0.9,
           yerr=stds, capsize=6, ecolor="#333333")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel(f"Throughput  (GFLOPS, {metric_txt(metric)})", fontsize=10)
    ax.set_title("Measured GFLOPS Throughput\n"
                 "GFLOPS = LEN×ITERS×2 / per-run seconds  ·  error bar = std dev"
                 "  ·  in-bar xN.NN = relative to fastest", fontsize=12)
    ax.grid(alpha=0.3, axis="y")
    annotate_bars(ax, xs, means, spec=".2f")

    # 柱身只标"相对最快者的倍数"(越大越好); 最快的那根直接标"最快"
    fastest = max(means)
    for x, y in zip(xs, means):
        mult = y / fastest
        ax.text(x, y * 0.55, "fastest" if mult >= 0.9999 else f"x{mult:.2f}",
                ha="center", va="center", fontsize=9.5, color="white", fontweight="bold")
    fig.suptitle(f"{args.csv_name}\n"
                 f"higher is better  ·  GFLOPS computed from each run's own CFG",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)
    print(f"[图4] GFLOPS 吞吐对比已保存: {out_path}")
    return True


def main(argv=None):
    p = argparse.ArgumentParser(description="把 run_bench.py 的 CSV 画成 4 张对比图")
    p.add_argument("csv", nargs="?", default=None, help="CSV 路径(默认取最新 bench_*.csv)")
    p.add_argument("--metric", default="wall", choices=["wall", "cpu"],
                   help="耗时口径(默认 wall=QPC 墙钟; cpu=QueryThreadCycleTime)")
    p.add_argument("--out-dir", default=None, help="输出目录(默认与 CSV 同目录)")
    p.add_argument("--prefix", default=None, help="输出文件名前缀(默认 CSV 文件名)")
    p.add_argument("--out", default=None,
                   help="兼容旧用法: 传 .png 路径时, 取其所在目录与文件名作为输出目录/前缀")
    p.add_argument("--y-scale", default="auto", choices=["auto", "log", "linear"],
                   help="图2 纵轴刻度(默认 auto: 动态范围 >30 倍自动取对数)")
    p.add_argument("--dpi", type=int, default=130, help="输出 PNG 分辨率(默认 130)")
    args = p.parse_args(argv)

    # ---- 定位 CSV ----
    if args.csv:
        csv_path = Path(args.csv)
        if not csv_path.is_absolute():
            csv_path = ROOT / csv_path
    else:
        cands = sorted(ROOT.glob("bench_*.csv"), key=lambda q: q.stat().st_mtime)
        if not cands:
            print("找不到 bench_*.csv; 请先跑 run_bench.py(默认会自动导出 CSV)",
                  file=sys.stderr)
            return 2
        csv_path = cands[-1]
    if not csv_path.exists():
        print(f"CSV 不存在: {csv_path}", file=sys.stderr)
        return 2

    # ---- 定位输出目录 / 文件名前缀 ----
    out_dir = csv_path.parent
    prefix = csv_path.stem
    if args.out:                       # 旧用法: --out some.png
        op = Path(args.out)
        out_dir = op.parent if str(op.parent) not in ("", ".") else ROOT
        if op.suffix.lower() == ".png":
            prefix = op.stem
    if args.out_dir:
        out_dir = Path(args.out_dir)
        if not out_dir.is_absolute():
            out_dir = ROOT / out_dir
    if args.prefix:
        prefix = args.prefix
    out_dir.mkdir(parents=True, exist_ok=True)
    args.csv_name = csv_path.name

    data = load_csv(csv_path)
    if not data:
        print(f"CSV 里没有有效样本: {csv_path}", file=sys.stderr)
        return 2

    sizes = sorted({k[1] for k in data})
    # ADD/MUL/SQRT 的内核变体只在图1 出现 —— 混进图2/3/4 会把曲线糊成一团
    kernel_impls = {n for _, simd_n, native_n in KERNEL_PAIRS for n in (simd_n, native_n)}
    known = [name for name, _, _ in IMPL_STYLE]
    impls = [n for n in known if any(k[0] == n for k in data)]
    impls += sorted({k[0] for k in data} - set(known) - kernel_impls)   # 未登记的实现也不丢

    print(f"CSV      : {csv_path}")
    print(f"尺寸     : {sizes}")
    print(f"实现     : {impls}")
    print(f"口径     : {args.metric}   输出目录: {out_dir}   前缀: {prefix}")

    made = []
    if fig_simd_vs_scalar(data, sizes, args, args.metric,
                          out_dir / f"{prefix}_1_simd_vs_scalar.png"):
        made.append(1)
    if fig_exec_time(data, sizes, impls, args, args.metric,
                     out_dir / f"{prefix}_2_exec_time.png"):
        made.append(2)
    if fig_overall_mean(data, sizes, impls, args, args.metric,
                        out_dir / f"{prefix}_3_overall_mean.png"):
        made.append(3)
    if fig_gflops(data, sizes, impls, args, args.metric,
                  out_dir / f"{prefix}_4_gflops.png"):
        made.append(4)

    # ---- 控制台再来一份文字小结, 免得每次都要开图 ----
    print(f"\n{'impl':<13} {'LEN':>9} {'工作集KiB':>10} {'每run us':>11} "
          f"{'GFLOPS':>8} {'CPU/墙钟':>9} {'trimmed':>8}")
    for s in sizes:
        for impl in impls:
            rec = data.get((impl, s))
            if not rec:
                continue
            wv = series(rec, "wall")
            cv = series(rec, "cpu")
            if wv.size == 0:
                continue
            flops_run = float(s) * float(rec["iters"]) * 2.0
            g = flops_run / (wv * 1e-6) / 1e9
            ratio = float(np.median(cv / wv[:cv.size])) if cv.size == wv.size else float("nan")
            print(f"{impl:<13} {s:>9} {s * 32 / 1024:>10.1f} {np.median(wv):>11.1f} "
                  f"{np.mean(g):>8.2f} {ratio:>9.4f} {rec['trimmed']:>8}")
    print(f"\n共生成 {len(made)} 张图: {made}")
    return 0 if made else 1


if __name__ == "__main__":
    sys.exit(main())
