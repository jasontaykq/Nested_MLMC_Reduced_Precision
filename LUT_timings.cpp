// LUT_timings.cpp  —  AVX-512 LUT Gaussian transforms: R repeats, 10M samples
//

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>

#include <immintrin.h>
#include <mkl.h>
#include <mkl_service.h>

#include "constant_lut_512.h"         
#include "avx512_lut_constant.h"      
#include "fp16x32.h"                  

#include "dyadic_lut.h"
#include "avx512_lut_dyadic.h"        

#include "superdyadic_lut_52.h"
#include "avx512_lut_superdyadic.h"  


#define N_EM_SAMPLES 10000000u
#define M 1

static inline double now_s() {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec*1e-9;
}
static inline void mean_sd(const std::vector<double>& v, double& mu, double& sd) {
    mu = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    double acc = 0.0; for (double x : v) acc += (x - mu) * (x - mu);
    sd = std::sqrt(acc / (v.size() - 1));
}
static inline double ns_per_sample(double seconds, size_t n) { return seconds * 1e9 / n; }

static inline void pre_touch(float* p, size_t n) {
    const size_t step = 4096 / sizeof(float);
    for (size_t i = 0; i < n; i += step) p[i] = 0.0f;
    if (n) p[n-1] = 0.0f;
}
static inline void pre_touch_h(_Float16* p, size_t n) {
    const size_t step = 4096 / sizeof(_Float16);
    for (size_t i = 0; i < n; i += step) p[i] = (_Float16)0.0f;
    if (n) p[n-1] = (_Float16)0.0f;
}


constexpr double T95_DF19 = 2.093;

int main() {
    mkl_set_num_threads_local(1);
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

    const int R = 20;
    const int WARMUP_GLOBAL = 3;
    const int WARMUP_PER_METHOD = 2;
    const size_t N = (size_t)M * N_EM_SAMPLES;

    // Buffers
    float*    all_u_f32              = (float*)    mkl_malloc(N * sizeof(float),    64);
    uint16_t* all_u_u16              = (uint16_t*) mkl_malloc(N * sizeof(uint16_t), 64);
    float*    z_const_f32            = (float*)    mkl_malloc(N * sizeof(float),    64);
    _Float16* z_const_f16            = (_Float16*) mkl_malloc(N * sizeof(_Float16), 64);
    float*    z_dyad_f32             = (float*)    mkl_malloc(N * sizeof(float),    64);
    _Float16* z_dyad_f16             = (_Float16*) mkl_malloc(N * sizeof(_Float16), 64);
    float*    z_super_f32            = (float*)    mkl_malloc(N * sizeof(float),    64);
    _Float16* z_super_f16            = (_Float16*) mkl_malloc(N * sizeof(_Float16), 64);
    if (!all_u_f32 || !all_u_u16 || !z_const_f32 || !z_const_f16 || !z_dyad_f32 || !z_dyad_f16 || !z_super_f32 || !z_super_f16) {
        fprintf(stderr, "Allocation failed\n"); return 1;
    }


    uint32_t* lut_fp16 = (uint32_t*) mkl_malloc(N_LUT * sizeof(uint32_t), 64);
    if (!lut_fp16) { fprintf(stderr, "Allocation failed (lut_fp16)\n"); return 1; }
    build_fp16_duplicated_lut(lut_fp16, GAUSS_LUT, N_LUT);


    prepare_polynomial_coefs();              
    prepare_polynomial_coefs_fp16_banks();   
    prepare_superdyadic_coefs_fp32();        
    prepare_superdyadic_coefs_fp16_banks();  


    VSLStreamStatePtr stream;
    vslNewStream(&stream, VSL_BRNG_MT19937, 12345); 

    //Warm up
    pre_touch(all_u_f32,     N);

    const size_t step_u16 = 4096 / sizeof(uint16_t);
    for (size_t i = 0; i < N; i += step_u16) all_u_u16[i] = 0;
    if (N) all_u_u16[N-1] = 0;
    pre_touch(z_const_f32,   N);
    pre_touch_h(z_const_f16, N);
    pre_touch(z_dyad_f32,    N);
    pre_touch_h(z_dyad_f16,  N);
    pre_touch(z_super_f32,   N);
    pre_touch_h(z_super_f16, N);

    for (int w = 0; w < WARMUP_GLOBAL; ++w) {
        vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, (MKL_INT)N, all_u_f32, 0.0f, 1.0f);
        for (size_t i = 0; i < N; ++i) {
            float u = all_u_f32[i];
            bool sign = (u >= 0.5f);
            float mag = sign ? (u - 0.5f) : u;
            _Float16 h = static_cast<_Float16>(mag);
            uint16_t bits;
            std::memcpy(&bits, &h, sizeof(uint16_t));
            all_u_u16[i] = (sign ? 0x8000u : 0u) | (bits & 0x7FFFu);
        }
        lut_piecewise_constant_fp32_avx512((unsigned)N, all_u_f32, z_const_f32);
    }

    // Per-repeat timings (seconds)
    std::vector<double> t_uni(R);
    std::vector<double> t_const32(R), t_const16(R);
    std::vector<double> t_dyad32(R),  t_dyad16(R);
    std::vector<double> t_super32(R), t_super16(R);

    // For speedups: per-repeat ratios FP32/FP16 (transform-only and end-to-end)
    std::vector<double> sp_const_trans(R), sp_dyad_trans(R), sp_super_trans(R);
    std::vector<double> sp_const_e2e(R),   sp_dyad_e2e(R),   sp_super_e2e(R);

    // Randomise method order per repeat to kill order bias (recommended by GPT)
    struct Entry { const char* name; int id; };
    std::vector<Entry> methods = {
        {"Constant-Uniform FP32", 0},
        {"Constant-Uniform FP16", 1},
        {"Linear-Dyadic   FP32",  2},
        {"Linear-Dyadic   FP16",  3},
        {"Linear-Super FP32",     4},
        {"Linear-Super FP16",     5}
    };
    std::mt19937 rng(2025);

    volatile double sink = 0.0;

    for (int r = 0; r < R; ++r) {
        double t0 = now_s();
        vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, (MKL_INT)N, all_u_f32, 0.0f, 1.0f);
        for (size_t i = 0; i < N; ++i) {
            float u = all_u_f32[i];
            bool sign = (u >= 0.5f);
            float mag = sign ? (u - 0.5f) : u;
            _Float16 h = static_cast<_Float16>(mag);
            uint16_t bits;
            std::memcpy(&bits, &h, sizeof(uint16_t));
            all_u_u16[i] = (sign ? 0x8000u : 0u) | (bits & 0x7FFFu);
        }
        double t1 = now_s();
        t_uni[r] = t1 - t0;

        // Shuffle method order this repeat
        std::shuffle(methods.begin(), methods.end(), rng);

        // Time each method (per-method warmups)
        for (const auto& m : methods) {
            for (int w = 0; w < WARMUP_PER_METHOD; ++w) {
                switch (m.id) {
                    case 0: lut_piecewise_constant_fp32_avx512((unsigned)N, all_u_f32, z_const_f32); break;
                    case 1: lut_piecewise_constant_fp16_avx512((unsigned)N, all_u_f32, z_const_f16, lut_fp16); break;
                    case 2: dyadic_lut_avx512_fp32((unsigned)N, all_u_u16, z_dyad_f32); break;
                    case 3: dyadic_lut_avx512_fp16((unsigned)N, all_u_u16, z_dyad_f16); break;
                    case 4: superdyadic_lut_avx512_fp32((unsigned)N, all_u_u16, z_super_f32); break;
                    case 5: superdyadic_lut_avx512_fp16((unsigned)N, all_u_u16, z_super_f16); break;
                }
            }
            // Timed call
            t0 = now_s();
            switch (m.id) {
                case 0: lut_piecewise_constant_fp32_avx512((unsigned)N, all_u_f32, z_const_f32); break;
                case 1: lut_piecewise_constant_fp16_avx512((unsigned)N, all_u_f32, z_const_f16, lut_fp16); break;
                case 2: dyadic_lut_avx512_fp32((unsigned)N, all_u_u16, z_dyad_f32); break;
                case 3: dyadic_lut_avx512_fp16((unsigned)N, all_u_u16, z_dyad_f16); break;
                case 4: superdyadic_lut_avx512_fp32((unsigned)N, all_u_u16, z_super_f32); break;
                case 5: superdyadic_lut_avx512_fp16((unsigned)N, all_u_u16, z_super_f16); break;
            }
            t1 = now_s();
            double dt = t1 - t0;

            // Record
            switch (m.id) {
                case 0: t_const32[r] = dt; sink += z_const_f32[0]; break;
                case 1: t_const16[r] = dt; sink += (double)z_const_f16[0]; break;
                case 2: t_dyad32[r]  = dt; sink += z_dyad_f32[1]; break;
                case 3: t_dyad16[r]  = dt; sink += (double)z_dyad_f16[1]; break;
                case 4: t_super32[r] = dt; sink += z_super_f32[2]; break;
                case 5: t_super16[r] = dt; sink += (double)z_super_f16[2]; break;
            }
        }


        auto nsps = [&](double sec){ return ns_per_sample(sec, N); };
        // Constant
        sp_const_trans[r] = nsps(t_const32[r]) / nsps(t_const16[r]);
        sp_const_e2e[r]   = nsps(t_const32[r] + t_uni[r]) / nsps(t_const16[r] + t_uni[r]);
        // Dyadic
        sp_dyad_trans[r]  = nsps(t_dyad32[r])  / nsps(t_dyad16[r]);
        sp_dyad_e2e[r]    = nsps(t_dyad32[r] + t_uni[r]) / nsps(t_dyad16[r] + t_uni[r]);
        // Super
        sp_super_trans[r] = nsps(t_super32[r]) / nsps(t_super16[r]);
        sp_super_e2e[r]   = nsps(t_super32[r] + t_uni[r]) / nsps(t_super16[r] + t_uni[r]);
    }


    auto print_stats = [&](const char* name,
                           const std::vector<double>& trans,
                           const std::vector<double>& uni) {
        double mu_t, sd_t, mu_e2e, sd_e2e;
        mean_sd(trans, mu_t, sd_t);
        std::vector<double> e2e(trans.size());
        for (size_t i = 0; i < trans.size(); ++i) e2e[i] = trans[i] + uni[i];
        mean_sd(e2e, mu_e2e, sd_e2e);

        printf("%-24s  transform-only: %.3f ± %.3f ns/sample   |   end-to-end: %.3f ± %.3f ns/sample\n",
               name,
               ns_per_sample(mu_t, N),   ns_per_sample(sd_t, N),
               ns_per_sample(mu_e2e, N), ns_per_sample(sd_e2e, N));
    };

    auto print_speedup = [&](const char* name,
                             const std::vector<double>& r_trans,
                             const std::vector<double>& r_e2e) {
        auto stats = [&](const std::vector<double>& r){
            double mu, sd; mean_sd(r, mu, sd);
            double se = sd / std::sqrt((double)r.size());
            double lo = mu - T95_DF19 * se;
            double hi = mu + T95_DF19 * se;
            return std::tuple<double,double,double,double>(mu, sd, lo, hi);
        };
        auto [mt, sdt, lot, hit] = stats(r_trans);
        auto [me, sde, loe, hie] = stats(r_e2e);
        printf("%-24s  speedup FP16/FP32  transform-only: %.2f× (95%% CI %.2f–%.2f)   |   end-to-end: %.2f× (%.2f–%.2f)\n",
               name, mt, lot, hit, me, loe, hie);
    };


    printf("\n=== Gaussian transform timings (R=%d, N=%zu per run) ===\n", R, N);
    print_stats("Constant-Uniform FP32", t_const32, t_uni);
    print_stats("Constant-Uniform FP16", t_const16, t_uni);
    print_stats("Linear-Dyadic   FP32", t_dyad32,  t_uni);
    print_stats("Linear-Dyadic   FP16", t_dyad16,  t_uni);
    print_stats("Linear-Super FP32",    t_super32, t_uni);
    print_stats("Linear-Super FP16",    t_super16, t_uni);

    printf("\n=== FP16 vs FP32 speedups (per-repeat ratios, mean ± 95%% CI) ===\n");
    print_speedup("Constant-Uniform", sp_const_trans, sp_const_e2e);
    print_speedup("Linear-Dyadic",    sp_dyad_trans,  sp_dyad_e2e);
    print_speedup("Linear-Super",     sp_super_trans, sp_super_e2e);

    // Cleanup
    vslDeleteStream(&stream);
    mkl_free(all_u_f32);
    mkl_free(all_u_u16);
    mkl_free(z_const_f32);
    mkl_free(z_const_f16);
    mkl_free(z_dyad_f32);
    mkl_free(z_dyad_f16);
    mkl_free(z_super_f32);
    mkl_free(z_super_f16);
    mkl_free(lut_fp16);

    return 0;
}

