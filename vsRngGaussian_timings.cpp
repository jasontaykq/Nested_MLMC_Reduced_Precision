// vsRngGaussian_timings.cpp  

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <ctime>
#include <vector>
#include <numeric>
#include <cmath>
#include <immintrin.h>
#include "mkl.h"
#include "mkl_service.h"

static inline double now_s() {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
static inline void mean_sd(const std::vector<double>& v, double& mu, double& sd) {
    mu = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    double acc = 0.0; for (double x : v) acc += (x - mu) * (x - mu);
    sd = std::sqrt(acc / (v.size() - 1));
}
static inline double ns_per_sample(double seconds, size_t n) { return seconds * 1e9 / n; }


static inline void f32_to_f16_avx512(const float* src, _Float16* dst, size_t n) {
    size_t i = 0;
#if defined(__AVX512FP16__)
    for (; i + 16 <= n; i += 16) {
        __m512  x   = _mm512_loadu_ps(src + i);
        __m256i xh  = _mm512_cvtps_ph(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm256_storeu_si256((__m256i*)(dst + i), xh);
    }
#endif
    for (; i < n; ++i) dst[i] = (_Float16)src[i];
}

int main() {
    const int    R        = 20;           // repeats
    const int    WARMUP   = 3;            // unrecorded runs
    const size_t nSamples = 10'000'000;   // 10M
    const int    baseSeed = 12345;        // deterministic + r for independence

    mkl_set_num_threads_local(1);         // single-threaded MKL


    float*    out_f32 = (float*)   mkl_malloc(nSamples * sizeof(float),   64);
    _Float16* out_f16 = (_Float16*) mkl_malloc(nSamples * sizeof(_Float16), 64);
    if (!out_f32 || !out_f16) { fprintf(stderr, "alloc failed\n"); return 1; }

    auto time_gauss_fp32 = [&](int method, int seed)->double {
        VSLStreamStatePtr s; vslNewStream(&s, VSL_BRNG_MT19937, seed);
        double t0 = now_s();
        vsRngGaussian(method, s, (MKL_INT)nSamples, out_f32, 0.0f, 1.0f);
        double t1 = now_s();
        vslDeleteStream(&s);
        return t1 - t0;
    };
    auto time_gauss_fp16 = [&](int method, int seed)->double {
        VSLStreamStatePtr s; vslNewStream(&s, VSL_BRNG_MT19937, seed);
        double t0 = now_s();
        vsRngGaussian(method, s, (MKL_INT)nSamples, out_f32, 0.0f, 1.0f);  // generate FP32
        f32_to_f16_avx512(out_f32, out_f16, nSamples);                      // convert to FP16
        double t1 = now_s();
        vslDeleteStream(&s);
        return t1 - t0;
    };

    // Methods to test
    struct M { const char* name; int method; };
    const M methods[] = {
        {"Box–Muller",   VSL_RNG_METHOD_GAUSSIAN_BOXMULLER},
        {"Box–Muller 2", VSL_RNG_METHOD_GAUSSIAN_BOXMULLER2},
        {"ICDF",         VSL_RNG_METHOD_GAUSSIAN_ICDF}
    };

    // Warmups (not recorded)
    for (int w = 0; w < WARMUP; ++w) {
        (void)time_gauss_fp32(VSL_RNG_METHOD_GAUSSIAN_ICDF, baseSeed + w);
    }

    // Timings
    for (const auto& m : methods) {
        std::vector<double> t32(R), t16(R);
        // repeats
        for (int r = 0; r < R; ++r) {
            t32[r] = time_gauss_fp32(m.method, baseSeed + r);
            t16[r] = time_gauss_fp16(m.method, baseSeed + 1000 + r);
        }
        double mu32, sd32, mu16, sd16;
        mean_sd(t32, mu32, sd32);
        mean_sd(t16, mu16, sd16);

        printf("\n=== %s (R=%d, N=%zu) ===\n", m.name, R, nSamples);
        printf("FP32: %.3f ± %.3f ns/sample\n", ns_per_sample(mu32, nSamples), ns_per_sample(sd32, nSamples));
        printf("FP16: %.3f ± %.3f ns/sample  (includes FP32→FP16 vectorised convert)\n",
               ns_per_sample(mu16, nSamples), ns_per_sample(sd16, nSamples));
    }

    mkl_free(out_f32);
    mkl_free(out_f16);
    return 0;
}

