/*
 * bench_c.c - 手写 C 基准实现 (进程内多次 run, 单进程输出多行)
 *
 * 内核(与 App/main.vp / bench_cpp.cpp / bench_py.py 一致):
 *   out[i] = a[i] * b[i] + c[i]     i = 0 .. LEN-1, LEN = 16384 (double)
 * 每次计时 run 连续执行 ITERS = 65536 趟内核, 单趟 2 次浮点运算/元素,
 * 即每次 run 总 flops = LEN*ITERS*2 ≈ 2.147e9。
 *
 * 命令行: bench_c.exe [runs]
 *   runs: 默认 1; 0 = 仅做热身(不输出 RESULT); 建议 10(由 run_bench.py 传入)。
 *
 * 流程:
 *   0) 入口先 SetThreadAffinityMask 绑到首个可用逻辑核(消除跨核迁移 / 大小核
 *      异构 / 缓存失效抖动), 并以 AFFINITY <core> 行回报(-1 = 绑定失败, 不致命);
 *   1) 进程开头打印一次 CFG <len> <iters>;
 *   2) 分配 a/b/c/out 并用固定种子 LCG 填充 a/b/c(不计时);
 *   3) 2 轮"满负荷热身"(每轮 = ITERS 趟内核 + 升序读回 out 求和), 之后再计时;
 *   4) 连续计时 runs 次, 每次输出一行:
 *        RESULT <wall_us> <cpu_us> <flops>
 *      wall_us = QPC 墙钟微秒(含被抢占 / 被降频的时间, 对调度敏感);
 *      cpu_us  = QueryThreadCycleTime 线程 CPU 微秒(只计本线程真正执行过的
 *                周期, 对抢占 / 降频不敏感; TSC 频率在入口用 QPC 标定);
 *      flops   = (LEN*ITERS*2)/秒, 按墙钟秒换算(与其它实现同口径);
 *      计时只包住 ITERS 趟内核本身(不含分配/填充/热身/读回校验);
 *   5) 全部结束后输出 CHECK <sum>(用最后一次 run 留下的 out 升序累加)。
 *
 * 防优化器删除 / 别名约束:
 *   - 所有指针一律加 __restrict(与 bench_cpp.cpp 同一拼写, clang/GCC/MSVC 都认):
 *     out 与 a/b/c 互不重叠, 关掉"保守别名分析"这个变量, 让 C 的自动向量化
 *     行为与 C++ 收敛(旧版 C 写 C99 的 restrict、C++ 写 __restrict, 两份源码
 *     拼写不一致, 容易把差异误判成语言差异);
 *   - 编译用 -O2 -mavx2 -mfma -ffp-contract=fast: 允许 FMA 收缩,
 *     out[i]=a[i]*b[i]+c[i] 生成单趟 vfmadd(单次舍入), 与 Viper 的
 *     simd_fma_array 同口径(不再强制"先乘后加"两次舍入);
 *   - 每轮热身与每次 run 之后都升序读回 out 求校验和 -> 计时区内的每一条
 *     out 写都"可观察", 不可能被整趟折叠/删除; noinline 防止内核被内联后
 *     与循环展开/标签冲突并杜绝跨趟优化。
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* QueryThreadCycleTime 是 Vista+ API: 显式抬到 0x0601 让 mingw 头文件声明它 */
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0601
#endif
#include <windows.h>

/* 负载约束: LEN × ITERS × 2 恒为 ≈2.147e9 flops/run(与其它实现可比),
 * 只调 LEN 来改"工作集"大小。
 * LEN=16384 -> a/b/c/out 四份共 512 KiB, 稳居本机(AMD Ryzen AI 7 H 350)
 * 每核 1 MiB 私有 L2 之内: 访存不再打到共享 L3 / DRAM, 也就不会被其它进程
 * 抢带宽 —— 这正是之前 C++ 出现"零星高时延尖峰"的主因。
 * 再往下压到 L1(需 LEN<=1024)会让每趟的内核调用/派发开销占比飙升,
 * 对 Python+numpy 尤其致命(实测每趟固定开销约 0.4us)。 */
#define LEN   16384           /* 元素数(2^14, 4 的倍数, 满足 ymm 4-lane 分批) */
#define ITERS 65536           /* 每 run 的内核趟数(令 LEN*ITERS*2 仍 ≈2.147e9) */
#define CFG_RUNS_MAX 200      /* runs 上限(防止误传超大值) */

#define LCG_A    48271.0
#define LCG_M    2147483647.0
#define LCG_SEED 12345.0

#if defined(_MSC_VER)
#define NOINLINE __declspec(noinline)
#else
#define NOINLINE __attribute__((noinline))
#endif

/* ------------------------------------------------------------
 * 线程绑定: 把当前线程绑到"第一个可用逻辑核", 返回该 CPU 序号; 失败返回 -1。
 * 从 1<<0 逐个试, 第一个被系统接受的位就是进程亲和掩码里编号最小的核 ——
 * 同一台机器上四个实现会绑到同一个核, 横向对比才有意义。
 * ------------------------------------------------------------ */
static int bind_first_available_core(void) {
    int i;
    const int nbits = (int)(sizeof(DWORD_PTR) * 8);
    for (i = 0; i < nbits; ++i) {
        if (SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 << i) != 0) {
            return i;
        }
    }
    return -1;
}

/* ------------------------------------------------------------
 * 线程 CPU 时间: QueryThreadCycleTime 返回本线程真正消耗过的 CPU 周期数
 * (现代 x86 的 TSC 是 invariant 的, 与核心实际频率无关)。
 *
 * 为什么不用 GetThreadTimes: 它的记账粒度跟随调度时钟节拍, 本机实测硬地板
 * 15.625ms —— 一个几十毫秒的 run 只能被记成 2~3 个节拍, 误差 ±40%,
 * 连"CPU 时间 > 墙钟时间"这种物理上不可能的结果都会出现。
 * 且 timeBeginPeriod(1) 在 Win10 2004+ 已无法改善该粒度(实测确认无效)。
 * QueryThreadCycleTime 的粒度是周期级, 彻底解决这个问题。
 * ------------------------------------------------------------ */
static long long query_thread_cycles(void) {
    ULONG64 cycles = 0;
    QueryThreadCycleTime(GetCurrentThread(), &cycles);
    return (long long)cycles;
}

/* 用 QPC 标定 TSC 频率(周期/秒): 忙等 30ms 数周期, 重复 3 次取最大。返回 0 = 失败。
 *
 * 为什么要取最大而不是平均: 标定窗口内本线程一旦被抢占, 消耗的周期数就少于
 * 墙钟经过的时间, 算出 hz 偏小 -> 后续 cpu_us 系统性偏大(实测约 +0.13%,
 * 会表现为"cpu_us 偶尔比 wall_us 还大"这种看着像 bug 的伪影)。取三次里的
 * 最大值 = 挑"最没被打断"的那次, 把该偏置压掉。
 * 必须在 SetThreadAffinityMask 之后调用, 否则跨核的 TSC 可能不同步。 */
static double calibrate_tsc_hz(LARGE_INTEGER freq) {
    double best = 0.0;
    int t;
    for (t = 0; t < 3; ++t) {
        LARGE_INTEGER a, b;
        long long c0, c1;
        double span = (double)freq.QuadPart * 0.030;   /* 30ms 对应的 QPC 计次 */
        double dt, hz;
        c0 = query_thread_cycles();
        QueryPerformanceCounter(&a);
        do {
            QueryPerformanceCounter(&b);
        } while ((double)(b.QuadPart - a.QuadPart) < span);
        c1 = query_thread_cycles();
        dt = (double)(b.QuadPart - a.QuadPart);
        if (dt <= 0.0 || c1 <= c0) {
            continue;
        }
        hz = (double)(c1 - c0) * (double)freq.QuadPart / dt;
        if (hz > best) {
            best = hz;
        }
    }
    return best;
}

/* 填充 a/b/c: 逐元素依次消耗同一个 LCG 流(a[i], b[i], c[i]) */
static void fill_abc(double *__restrict a,
                     double *__restrict b,
                     double *__restrict c,
                     long n) {
    double state = LCG_SEED;
    const double m = LCG_M;
    const double mul = LCG_A;
    long i;
    for (i = 0; i < n; ++i) {
        state = fmod(state * mul, m);
        a[i] = state / m;
        state = fmod(state * mul, m);
        b[i] = state / m;
        state = fmod(state * mul, m);
        c[i] = state / m;
    }
}

/* 内核: out[i] = a[i]*b[i] + c[i] (元素间独立, -O2 + AVX2 + FMA 自动向量化单趟融合乘加)
 * __restrict: 四个指针互不别名, 编译器无需为"可能重叠"生成保守标量路径。 */
static NOINLINE void kernel_fma(double *__restrict out,
                                const double *__restrict a,
                                const double *__restrict b,
                                const double *__restrict c,
                                long n) {
    long i;
    for (i = 0; i < n; ++i) {
        out[i] = a[i] * b[i] + c[i];
    }
}

/* 校验和: 升序逐元素累加 out(读回 out -> 保证内核的写未被删除) */
static double checksum_sum(const double *__restrict out, long n) {
    double s = 0.0;
    long i;
    for (i = 0; i < n; ++i) {
        s += out[i];
    }
    return s;
}

int main(int argc, char **argv) {
    /* 命令行: bench_c.exe [runs] [len] [iters]
     *   省略则用上面 #define 的默认值; run_bench.py 做多尺寸扫描时会显式传入。 */
    long n = LEN;
    long iters = ITERS;
    size_t nb;
    double total_flops;
    /* 四个缓冲各自独立, 指针全部 __restrict: 与内核形参同一 no-alias 承诺 */
    double *__restrict a, *__restrict b, *__restrict c, *__restrict out;
    LARGE_INTEGER freq, t0, t1;
    double last_chk = 0.0;
    double tsc_hz;
    int runs = 1;
    int r;
    int core;
    long k;

    if (argc > 1) {
        runs = atoi(argv[1]);
    }
    if (argc > 2) {
        n = atol(argv[2]);
    }
    if (argc > 3) {
        iters = atol(argv[3]);
    }
    if (runs < 0) runs = 0;
    if (runs > CFG_RUNS_MAX) runs = CFG_RUNS_MAX;
    /* 内核按 4-lane ymm 分批, LEN 必须是 4 的倍数; 参数非法直接报错, 不静默修正 */
    if (n < 4 || n % 4 != 0 || iters < 1) {
        fprintf(stderr, "bad args: len=%ld(需 >=4 且为 4 的倍数) iters=%ld\n", n, iters);
        return 2;
    }
    nb = (size_t)n * sizeof(double);
    total_flops = (double)n * (double)iters * 2.0;

    /* 线程绑定(必做) + 提优先级: 消除跨核迁移, 并压低背景进程抢占造成的零星尖峰 */
    core = bind_first_available_core();
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);

    printf("CFG %ld %ld\n", n, iters);
    printf("AFFINITY %d\n", core);
    fflush(stdout);

    a = (double *)malloc(nb);
    b = (double *)malloc(nb);
    c = (double *)malloc(nb);
    out = (double *)malloc(nb);
    if (!a || !b || !c || !out) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }

    /* 输入生成一次(不计时) */
    fill_abc(a, b, c, n);

    QueryPerformanceFrequency(&freq);

    /* TSC 频率标定(必须在绑核之后): 把 QueryThreadCycleTime 的周期数换算成微秒 */
    tsc_hz = calibrate_tsc_hz(freq);
    printf("TSC %.4e\n", tsc_hz);
    fflush(stdout);

    /* 2 轮满负荷热身(每轮 = ITERS 趟内核 + 读回 out 求校验和) */
    for (r = 0; r < 2; ++r) {
        for (k = 0; k < iters; ++k) {
            kernel_fma(out, a, b, c, n);
        }
        last_chk = checksum_sum(out, n);
    }

    /* 连续计时 runs 次; 计时只包住 ITERS 趟内核
     * 双计时: 墙钟 QPC + 线程 CPU(QueryThreadCycleTime), 同时包住同一段内核 */
    for (r = 0; r < runs; ++r) {
        double us, cpu_us, flops;
        long long cy0, cy1;
        cy0 = query_thread_cycles();
        QueryPerformanceCounter(&t0);
        for (k = 0; k < iters; ++k) {
            kernel_fma(out, a, b, c, n);
        }
        QueryPerformanceCounter(&t1);
        cy1 = query_thread_cycles();
        us = ((double)(t1.QuadPart - t0.QuadPart) * 1000000.0) /
             (double)freq.QuadPart;
        /* 周期 -> 微秒; 标定失败(tsc_hz==0)时给 0, 表现为一眼可见的异常 */
        cpu_us = (tsc_hz > 0.0) ? (double)(cy1 - cy0) * 1000000.0 / tsc_hz : 0.0;
        flops = total_flops / (us * 1e-6);
        printf("RESULT %.3f %.3f %.0f\n", us, cpu_us, flops);
        fflush(stdout);
        /* 计时区外读回 out: 使本轮写入可观察, 阻止跨 run 的写折叠 */
        last_chk = checksum_sum(out, n);
    }

    printf("CHECK %.17g\n", last_chk);
    fflush(stdout);

    free(a);
    free(b);
    free(c);
    free(out);
    return 0;
}
