/*
 * bench_cpp.cpp - 手写 C++ 基准实现 (进程内多次 run, 单进程输出多行)
 *
 * 与 bench_c.c 完全同口径:
 *   内核   out[i] = a[i]*b[i] + c[i], LEN = 16384, 每次 run = ITERS = 65536 趟,
 *   总 flops ≈ 2.147e9; 命令行第 1 参数 = runs(默认 1; 0 = 仅热身)。
 *
 * - 容器 std::vector; 内核经裸指针 + __restrict 传入, 便于自动向量化。
 * - 线程绑定: 入口 SetThreadAffinityMask 绑到首个可用逻辑核, 以 AFFINITY <core>
 *   行回报(-1 = 绑定失败)。与 bench_c.c / main.vp / bench_py.py 绑同一个核。
 * - 双计时: 每次 run 同时采集 QPC 墙钟微秒 与 QueryThreadCycleTime 线程 CPU
 *   微秒, 输出 RESULT <wall_us> <cpu_us> <flops>(flops 按墙钟秒换算)。
 * - 随机输入与 CHECK 口径与 C/Viper/Python 完全一致(固定种子 minstd LCG)。
 * - 编译口径与 bench_c.c 相同: -O2 -mavx2 -mfma -ffp-contract=fast,
 *   允许 FMA 收缩成单趟 vfmadd(单次舍入), 与 Viper 的 simd_fma_array 对齐。
 * - 防删: 每轮热身/每次 run 后升序读回 out 求 CHECK; noinline; __restrict。
 */
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

/* QueryThreadCycleTime 是 Vista+ API: 显式抬到 0x0601 让 mingw 头文件声明它 */
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0601
#endif
#include <windows.h>

namespace {

/* 负载约束: N_ELEM × N_ITERS × 2 恒为 ≈2.147e9 flops/run(与其它实现可比),
 * 只调 N_ELEM 来改"工作集"大小。
 * N_ELEM=16384 -> a/b/c/out 四份共 512 KiB, 稳居本机每核 1 MiB 私有 L2 之内:
 * 访存不再打到共享 L3 / DRAM, 也就不会被其它进程抢带宽 —— 这正是之前 C++
 * 出现"零星高时延尖峰"的主因(与 std::vector/向量化无关, 计时区内核逐字相同)。 */
constexpr long N_ELEM = 16384;
constexpr long N_ITERS = 65536;
constexpr int RUNS_MAX = 200;

constexpr double LCG_A = 48271.0;
constexpr double LCG_M = 2147483647.0;
constexpr double LCG_SEED = 12345.0;

/* ------------------------------------------------------------
 * 线程绑定: 绑到"第一个可用逻辑核", 返回 CPU 序号; 失败返回 -1。
 * 从 1<<0 逐个试, 第一个被接受的位就是进程亲和掩码里编号最小的核 ——
 * 四个实现绑到同一个核, 横向对比才有意义。
 * ------------------------------------------------------------ */
int bind_first_available_core() {
    const int nbits = static_cast<int>(sizeof(DWORD_PTR) * 8);
    for (int i = 0; i < nbits; ++i) {
        if (SetThreadAffinityMask(GetCurrentThread(),
                                  static_cast<DWORD_PTR>(1) << i) != 0) {
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
 * 15.625ms —— 几十毫秒的 run 只能被记成 2~3 个节拍(误差 ±40%), 连
 * "CPU 时间 > 墙钟时间"这种物理上不可能的结果都会出现; 且
 * timeBeginPeriod(1) 在 Win10 2004+ 已无法改善该粒度(实测确认无效)。
 * ------------------------------------------------------------ */
long long query_thread_cycles() {
    ULONG64 cycles = 0;
    QueryThreadCycleTime(GetCurrentThread(), &cycles);
    return static_cast<long long>(cycles);
}

/* 用 QPC 标定 TSC 频率(周期/秒): 忙等 30ms 数周期, 重复 3 次取最大。0.0 = 失败。
 *
 * 为什么取最大: 标定窗口内本线程一旦被抢占, 消耗周期就少于墙钟流逝时间,
 * 算出 hz 偏小 -> 后续 cpu_us 系统性偏大(实测约 +0.13%, 表现为"cpu_us 偶尔
 * 比 wall_us 还大"这种看着像 bug 的伪影)。取三次最大值 = 挑最没被打断的一次。
 * 必须在 SetThreadAffinityMask 之后调用, 否则跨核的 TSC 可能不同步。 */
double calibrate_tsc_hz(const LARGE_INTEGER &freq) {
    const double span = static_cast<double>(freq.QuadPart) * 0.030;  /* 30ms 的 QPC 计次 */
    double best = 0.0;
    for (int t = 0; t < 3; ++t) {
        LARGE_INTEGER a{}, b{};
        const long long c0 = query_thread_cycles();
        QueryPerformanceCounter(&a);
        do {
            QueryPerformanceCounter(&b);
        } while (static_cast<double>(b.QuadPart - a.QuadPart) < span);
        const long long c1 = query_thread_cycles();
        const double dt = static_cast<double>(b.QuadPart - a.QuadPart);
        if (dt <= 0.0 || c1 <= c0) {
            continue;
        }
        const double hz =
            static_cast<double>(c1 - c0) * static_cast<double>(freq.QuadPart) / dt;
        if (hz > best) {
            best = hz;
        }
    }
    return best;
}

#if defined(_MSC_VER)
#define NOINLINE __declspec(noinline)
#else
#define NOINLINE __attribute__((noinline))
#endif

/* 填充 a/b/c: 逐元素依次消耗同一个 LCG 流(a[i], b[i], c[i]) */
void fill_abc(double *a, double *b, double *c, long n) {
    double state = LCG_SEED;
    const double m = LCG_M;
    const double mul = LCG_A;
    for (long i = 0; i < n; ++i) {
        state = std::fmod(state * mul, m);
        a[i] = state / m;
        state = std::fmod(state * mul, m);
        b[i] = state / m;
        state = std::fmod(state * mul, m);
        c[i] = state / m;
    }
}

/* 内核: out[i] = a[i]*b[i] + c[i] (元素间独立, -O2 + AVX2 + FMA 自动向量化单趟融合乘加) */
static NOINLINE void kernel_fma(double *__restrict out,
                                const double *__restrict a,
                                const double *__restrict b,
                                const double *__restrict c,
                                long n) {
    for (long i = 0; i < n; ++i) {
        out[i] = a[i] * b[i] + c[i];
    }
}

/* 校验和: 升序逐元素累加 out(读回 out -> 保证内核的写未被删除) */
double checksum_sum(const double *__restrict out, long n) {
    double s = 0.0;
    for (long i = 0; i < n; ++i) {
        s += out[i];
    }
    return s;
}

}  // namespace

int main(int argc, char **argv) {
    /* 命令行: bench_cpp.exe [runs] [len] [iters]
     *   省略则用上面 constexpr 的默认值; run_bench.py 多尺寸扫描时会显式传入。 */
    long n = N_ELEM;
    long iters = N_ITERS;
    int runs = 1;
    if (argc > 1) {
        runs = std::atoi(argv[1]);
    }
    if (argc > 2) {
        n = std::atol(argv[2]);
    }
    if (argc > 3) {
        iters = std::atol(argv[3]);
    }
    if (runs < 0) runs = 0;
    if (runs > RUNS_MAX) runs = RUNS_MAX;
    /* 内核按 4-lane ymm 分批, LEN 必须是 4 的倍数; 参数非法直接报错, 不静默修正 */
    if (n < 4 || n % 4 != 0 || iters < 1) {
        std::fprintf(stderr, "bad args: len=%ld(需 >=4 且为 4 的倍数) iters=%ld\n", n, iters);
        return 2;
    }
    const double total_flops = static_cast<double>(n) * static_cast<double>(iters) * 2.0;

    /* 线程绑定(必做) + 提优先级: 消除跨核迁移, 并压低背景进程抢占造成的零星尖峰 */
    const int core = bind_first_available_core();
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);

    std::printf("CFG %ld %ld\n", n, iters);
    std::printf("AFFINITY %d\n", core);
    std::fflush(stdout);

    std::vector<double> a(n), b(n), c(n), out(n);
    double *pa = a.data();
    double *pb = b.data();
    double *pc = c.data();
    double *pout = out.data();

    LARGE_INTEGER freq, t0, t1;
    double last_chk = 0.0;

    /* 输入生成一次(不计时) */
    fill_abc(pa, pb, pc, n);

    QueryPerformanceFrequency(&freq);

    /* TSC 频率标定(必须在绑核之后): 把 QueryThreadCycleTime 的周期数换算成微秒 */
    const double tsc_hz = calibrate_tsc_hz(freq);
    std::printf("TSC %.4e\n", tsc_hz);
    std::fflush(stdout);

    /* 2 轮满负荷热身(每轮 = ITERS 趟内核 + 读回 out 求校验和) */
    for (int r = 0; r < 2; ++r) {
        for (long k = 0; k < iters; ++k) {
            kernel_fma(pout, pa, pb, pc, n);
        }
        last_chk = checksum_sum(pout, n);
    }

    /* 连续计时 runs 次; 计时只包住 ITERS 趟内核
     * 双计时: 墙钟 QPC + 线程 CPU(QueryThreadCycleTime), 同时包住同一段内核 */
    for (int r = 0; r < runs; ++r) {
        const long long cy0 = query_thread_cycles();
        QueryPerformanceCounter(&t0);
        for (long k = 0; k < iters; ++k) {
            kernel_fma(pout, pa, pb, pc, n);
        }
        QueryPerformanceCounter(&t1);
        const long long cy1 = query_thread_cycles();

        const double us =
            (static_cast<double>(t1.QuadPart - t0.QuadPart) * 1000000.0) /
            static_cast<double>(freq.QuadPart);
        /* 周期 -> 微秒; 标定失败(tsc_hz==0)时给 0, 表现为一眼可见的异常 */
        const double cpu_us = (tsc_hz > 0.0)
            ? static_cast<double>(cy1 - cy0) * 1000000.0 / tsc_hz : 0.0;
        const double flops = total_flops / (us * 1e-6);
        std::printf("RESULT %.3f %.3f %.0f\n", us, cpu_us, flops);
        std::fflush(stdout);
        /* 计时区外读回 out: 阻止跨 run 的写折叠 */
        last_chk = checksum_sum(pout, n);
    }

    std::printf("CHECK %.17g\n", last_chk);
    std::fflush(stdout);
    return 0;
}
