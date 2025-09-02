//payoff_path_timings_mlmc.cpp
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <immintrin.h>
#include <mkl.h>
#include <algorithm>
#include <vector>


#define N_SAMPLES 1000000u  // 1 million samples per level for timing
#define R         20         // Number of repetitions
#define WARMUP    3          // Warmup runs

// MLMC levels to test
const std::vector<int> MLMC_LEVELS = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512};

// Timing helper
static inline double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Vector widths
static const unsigned int V_F32 = 16u;
static const unsigned int V_F16 = 32u;

// FP32 AVX-512 kernel
void compute_fp32_payoffs(
    const float* all_z,
    float* out_payoffs,
    unsigned int N_samples,
    int M_STEPS,
    float S0,
    float K,
    float r,
    float sigma,
    float sqrt_h,
    float discount_factor)
{
    __m512 r_h          = _mm512_set1_ps(r * (sqrt_h*sqrt_h));
    __m512 sigma_sqrt_h = _mm512_set1_ps(sigma * sqrt_h);
    __m512 one_v        = _mm512_set1_ps(1.0f);
    __m512 K_v          = _mm512_set1_ps(K);
    __m512 disc_v       = _mm512_set1_ps(discount_factor);

    unsigned int i = 0;
    for (; i + V_F32 <= N_samples; i += V_F32) {
        __m512 S = _mm512_set1_ps(S0);
        for (int step = 0; step < M_STEPS; ++step) {
            const float* z_ptr = all_z + step * N_samples + i;
            __m512 z = _mm512_loadu_ps(z_ptr);
            __m512 tmp = _mm512_fmadd_ps(sigma_sqrt_h, z, r_h);
            tmp = _mm512_add_ps(tmp, one_v);
            S = _mm512_mul_ps(S, tmp);
        }
        __m512 diff = _mm512_sub_ps(S, K_v);
        __mmask16 m  = _mm512_cmp_ps_mask(diff, _mm512_setzero_ps(), _CMP_GT_OS);
        __m512 payoff = _mm512_mask_mul_ps(_mm512_setzero_ps(), m, diff, disc_v);
        _mm512_storeu_ps(out_payoffs + i, payoff);
    }
    for (; i < N_samples; ++i) {
        float S = S0;
        for (int step = 0; step < M_STEPS; ++step) {
            float z = all_z[step * N_samples + i];
            S *= (1.0f + r * (sqrt_h*sqrt_h) + sigma * sqrt_h * z);
        }
        out_payoffs[i] = std::max(S - K, 0.0f) * discount_factor;
    }
}

// FP16 AVX-512 kernel with pure FP16 arithmetic
void compute_fp16_payoffs(
    const _Float16* all_z,
    _Float16* out_payoffs,
    unsigned int N_samples,
    int M_STEPS,
    _Float16 S0,
    _Float16 K,
    _Float16 r,
    _Float16 sigma,
    _Float16 sqrt_h,
    _Float16 discount_factor)
{
    // Precompute constants in FP16
    _Float16 h = sqrt_h * sqrt_h;
    _Float16 r_h = r * h;
    _Float16 sigma_sqrt_h = sigma * sqrt_h;
    _Float16 one = (_Float16)1.0;
    
    // Load constants into FP16 vectors
    __m512h S0_v       = _mm512_set1_ph(S0);
    __m512h K_v        = _mm512_set1_ph(K);
    __m512h r_h_v      = _mm512_set1_ph(r_h);
    __m512h sigma_sqrt = _mm512_set1_ph(sigma_sqrt_h);
    __m512h one_v      = _mm512_set1_ph(one);
    __m512h disc_v     = _mm512_set1_ph(discount_factor);
    __m512h zero_v     = _mm512_setzero_ph();

    unsigned int i = 0;
    for (; i + V_F16 <= N_samples; i += V_F16) {
        __m512h S = S0_v;
        
        for (int step = 0; step < M_STEPS; ++step) {
            const _Float16* z_ptr = all_z + step * N_samples + i;
            __m512h z = _mm512_loadu_ph(z_ptr);
            __m512h tmp = _mm512_fmadd_ph(sigma_sqrt, z, r_h_v);
            tmp = _mm512_add_ph(tmp, one_v);
            S = _mm512_mul_ph(S, tmp);
        }
        
        __m512h diff = _mm512_sub_ph(S, K_v);
        __mmask32 m = _mm512_cmp_ph_mask(diff, zero_v, _CMP_GT_OS);
        __m512h payoff = _mm512_mask_mul_ph(zero_v, m, diff, disc_v);
        _mm512_storeu_ph(out_payoffs + i, payoff);
    }
    
    for (; i < N_samples; ++i) {
        _Float16 S = S0;
        for (int step = 0; step < M_STEPS; ++step) {
            _Float16 z = all_z[step * N_samples + i];
            S = S * (one + r_h + sigma_sqrt_h * z);
        }
        _Float16 diff = S - K;
        out_payoffs[i] = (diff > 0) ? diff * discount_factor : (_Float16)0;
    }
}

struct LevelTiming {
    int steps;
    double h;
    double fp32_time_mean;
    double fp32_time_stddev;
    double fp16_time_mean;
    double fp16_time_stddev;
    double rng_time_mean;
    double rng_time_stddev;
};

struct GaussianData {
    float* f32_samples;
    _Float16* f16_samples;
    double generation_time;
};


GaussianData generate_gaussian_samples(int n_samples, int m_steps, VSLStreamStatePtr stream) {
    GaussianData data;
    data.f32_samples = (float*)mkl_malloc(n_samples * m_steps * sizeof(float), 64);
    data.f16_samples = (_Float16*)mkl_malloc(n_samples * m_steps * sizeof(_Float16), 64);
    
    double start_time = get_time();
    for (int step = 0; step < m_steps; ++step) {
        vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, stream,
                     n_samples, data.f32_samples + step * n_samples, 0.0f, 1.0f);
        for (int i = 0; i < n_samples; ++i) {
            data.f16_samples[step * n_samples + i] = 
                static_cast<_Float16>(data.f32_samples[step * n_samples + i]);
        }
    }
    data.generation_time = (get_time() - start_time) * 1e6;
    
    return data;
}

void compute_time_stats(const std::vector<double>& times, double& mean, double& stddev) {
    mean = 0.0;
    for (double t : times) mean += t;
    mean /= times.size();
    
    stddev = 0.0;
    for (double t : times) stddev += (t - mean) * (t - mean);
    stddev = std::sqrt(stddev / times.size());
}

// Time a single level
LevelTiming time_mlmc_level(int m_steps, int n_samples, VSLStreamStatePtr stream) {
    const float T = 1.0f;
    const float h = T / m_steps;
    const float sqrt_h = std::sqrt(h);
    const float r = 0.05f;
    const float sigma = 0.2f;
    const float S0_f32 = 100.0f;
    const float K_f32 = 100.0f;
    const _Float16 r_f16 = (_Float16)r;
    const _Float16 sigma_f16 = (_Float16)sigma;
    const _Float16 sqrt_h_f16 = (_Float16)sqrt_h;
    const _Float16 S0_f16 = (_Float16)S0_f32;
    const _Float16 K_f16 = (_Float16)K_f32;
    float discount_f32 = std::exp(-r * T);
    _Float16 discount_f16 = (_Float16)discount_f32;

    float* payoff_f32 = (float*)mkl_malloc(n_samples * sizeof(float), 64);
    _Float16* payoff_f16 = (_Float16*)mkl_malloc(n_samples * sizeof(_Float16), 64);

    std::vector<double> fp32_times(R);
    std::vector<double> fp16_times(R);
    std::vector<double> rng_times(R);

    // Warmup
    for (int warmup = 0; warmup < WARMUP; ++warmup) {
        auto gaussian_data = generate_gaussian_samples(n_samples, m_steps, stream);
        compute_fp32_payoffs(gaussian_data.f32_samples, payoff_f32, n_samples, 
                           m_steps, S0_f32, K_f32, r, sigma, sqrt_h, discount_f32);
        compute_fp16_payoffs(gaussian_data.f16_samples, payoff_f16, n_samples,
                           m_steps, S0_f16, K_f16, r_f16, sigma_f16, sqrt_h_f16, discount_f16);
        mkl_free(gaussian_data.f32_samples);
        mkl_free(gaussian_data.f16_samples);
    }

    // Timed runs
    for (int run = 0; run < R; ++run) {
        auto gaussian_data = generate_gaussian_samples(n_samples, m_steps, stream);
        rng_times[run] = gaussian_data.generation_time;

        double start_time = get_time();
        compute_fp32_payoffs(gaussian_data.f32_samples, payoff_f32, n_samples,
                           m_steps, S0_f32, K_f32, r, sigma, sqrt_h, discount_f32);
        double mid_time = get_time();
        compute_fp16_payoffs(gaussian_data.f16_samples, payoff_f16, n_samples,
                           m_steps, S0_f16, K_f16, r_f16, sigma_f16, sqrt_h_f16, discount_f16);
        double end_time = get_time();

        fp32_times[run] = (mid_time - start_time) * 1e6;
        fp16_times[run] = (end_time - mid_time) * 1e6;

        mkl_free(gaussian_data.f32_samples);
        mkl_free(gaussian_data.f16_samples);
    }

    LevelTiming result;
    result.steps = m_steps;
    result.h = h;
    compute_time_stats(fp32_times, result.fp32_time_mean, result.fp32_time_stddev);
    compute_time_stats(fp16_times, result.fp16_time_mean, result.fp16_time_stddev);
    compute_time_stats(rng_times, result.rng_time_mean, result.rng_time_stddev);

    mkl_free(payoff_f32);
    mkl_free(payoff_f16);

    return result;
}

int main() {
    mkl_set_num_threads_local(1);
    VSLStreamStatePtr stream;
    vslNewStream(&stream, VSL_BRNG_MT19937, 12345);

    printf("MLMC Timing Analysis: Levels 1 to 64\n");
    printf("Samples per level: %u, Repetitions: %d\n\n", N_SAMPLES, R);

    std::vector<LevelTiming> results;

    for (int m_steps : MLMC_LEVELS) {
        printf("Testing level M=%d...\n", m_steps);
        LevelTiming timing = time_mlmc_level(m_steps, N_SAMPLES, stream);
        results.push_back(timing);
    }
    
        // Print results table
    printf("\n=== MLMC TIMING RESULTS (mean ± SD in microseconds) ===\n");
    printf("%-6s %-8s %-12s %-12s %-12s %-8s %-8s\n", 
           "M", " h ", " FP32 Time ", " FP16 Time ", " RNG Time ", "  Speedup  ", "  Scaling  ");
    printf("-------------------------------------------------------------\n");

    double base_fp32 = results[0].fp32_time_mean;
    double base_fp16 = results[0].fp16_time_mean;

    for (const auto& res : results) {
        double speedup = res.fp32_time_mean / res.fp16_time_mean;
        double scaling = res.fp32_time_mean / base_fp32;
        double expected_scaling = res.steps;

        printf("%-6d %-8.4f  |  %-6.1f±%-4.1f  |  %-6.1f±%-4.1f  |  %-6.1f±%-4.1f  |  %-8.2fx  |  %-8.2fx\n",
               res.steps, res.h,
               res.fp32_time_mean, res.fp32_time_stddev,
               res.fp16_time_mean, res.fp16_time_stddev,
               res.rng_time_mean, res.rng_time_stddev,
               speedup, scaling);
    }


    
// --- PER-SAMPLE & PER-PATH-STEP (ns; mean ± SD over R runs) ---
printf("\n=== PER-SAMPLE & PER-PATH-STEP TIMING (ns; mean \u00B1 SD over %d runs) ===\n", R);
printf("%-6s %-22s %-22s %-22s | %-26s %-26s %-26s\n",
       "M", "FP16/Path", "FP32/Path", "RNG/Path",
           "FP16/Path-Step", "FP32/Path-Step", "RNG/Path-Step");
printf("---------------------------------------------------------------------------------------------------------------\n");

for (const auto& res : results) {
    // per-path (divide by N)
    double fp32_path_mean = (res.fp32_time_mean   * 1000.0) / (double)N_SAMPLES;
    double fp32_path_sd   = (res.fp32_time_stddev * 1000.0) / (double)N_SAMPLES;
    double fp16_path_mean = (res.fp16_time_mean   * 1000.0) / (double)N_SAMPLES;
    double fp16_path_sd   = (res.fp16_time_stddev * 1000.0) / (double)N_SAMPLES;
    double rng_path_mean  = (res.rng_time_mean    * 1000.0) / (double)N_SAMPLES;
    double rng_path_sd    = (res.rng_time_stddev  * 1000.0) / (double)N_SAMPLES;

    // per-path-step (divide by N*M)
    double denom = (double)N_SAMPLES * (double)res.steps;
    double fp32_ps_mean = (res.fp32_time_mean   * 1000.0) / denom;
    double fp32_ps_sd   = (res.fp32_time_stddev * 1000.0) / denom;
    double fp16_ps_mean = (res.fp16_time_mean   * 1000.0) / denom;
    double fp16_ps_sd   = (res.fp16_time_stddev * 1000.0) / denom;
    double rng_ps_mean  = (res.rng_time_mean    * 1000.0) / denom;
    double rng_ps_sd    = (res.rng_time_stddev  * 1000.0) / denom;

    printf("%-6d %8.3f \u00B1 %-9.3f %8.3f \u00B1 %-9.3f %8.3f \u00B1 %-9.3f | %9.3f \u00B1 %-11.3f %9.3f \u00B1 %-11.3f %9.3f \u00B1 %-11.3f\n",
           res.steps,
           fp16_path_mean, fp16_path_sd,
           fp32_path_mean, fp32_path_sd,
           rng_path_mean,  rng_path_sd,
           fp16_ps_mean,   fp16_ps_sd,
           fp32_ps_mean,   fp32_ps_sd,
           rng_ps_mean,    rng_ps_sd);
}

{
    FILE* csv = fopen("mlmc_path_timings.csv", "w");
    if (!csv) {
        perror("Failed to open mlmc_timings.csv");
    } else {
        // Header
        fprintf(csv,
            "M,h,"
            "FP16_Path_mean_ns,FP16_Path_sd_ns,"
            "FP32_Path_mean_ns,FP32_Path_sd_ns,"
            "RNG_Path_mean_ns,RNG_Path_sd_ns,"
            "FP16_PathStep_mean_ns,FP16_PathStep_sd_ns,"
            "FP32_PathStep_mean_ns,FP32_PathStep_sd_ns,"
            "RNG_PathStep_mean_ns,RNG_PathStep_sd_ns\n");

        for (const auto& res : results) {
            // per-path (ns)
            double fp32_path_mean = (res.fp32_time_mean   * 1000.0) / (double)N_SAMPLES;
            double fp32_path_sd   = (res.fp32_time_stddev * 1000.0) / (double)N_SAMPLES;
            double fp16_path_mean = (res.fp16_time_mean   * 1000.0) / (double)N_SAMPLES;
            double fp16_path_sd   = (res.fp16_time_stddev * 1000.0) / (double)N_SAMPLES;
            double rng_path_mean  = (res.rng_time_mean    * 1000.0) / (double)N_SAMPLES;
            double rng_path_sd    = (res.rng_time_stddev  * 1000.0) / (double)N_SAMPLES;

            // per-path-step (ns)
            double denom = (double)N_SAMPLES * (double)res.steps;
            double fp32_ps_mean = (res.fp32_time_mean   * 1000.0) / denom;
            double fp32_ps_sd   = (res.fp32_time_stddev * 1000.0) / denom;
            double fp16_ps_mean = (res.fp16_time_mean   * 1000.0) / denom;
            double fp16_ps_sd   = (res.fp16_time_stddev * 1000.0) / denom;
            double rng_ps_mean  = (res.rng_time_mean    * 1000.0) / denom;
            double rng_ps_sd    = (res.rng_time_stddev  * 1000.0) / denom;

            fprintf(csv,
                "%d,%.10f,"
                "%.6f,%.6f,"
                "%.6f,%.6f,"
                "%.6f,%.6f,"
                "%.6f,%.6f,"
                "%.6f,%.6f,"
                "%.6f,%.6f\n",
                res.steps, res.h,
                fp16_path_mean, fp16_path_sd,
                fp32_path_mean, fp32_path_sd,
                rng_path_mean,  rng_path_sd,
                fp16_ps_mean,   fp16_ps_sd,
                fp32_ps_mean,   fp32_ps_sd,
                rng_ps_mean,    rng_ps_sd
            );
        }
        fclose(csv);
        printf("\nSaved CSV: mlmc_path_timings.csv\n");
    }
}

    vslDeleteStream(&stream);
    return 0;
}




