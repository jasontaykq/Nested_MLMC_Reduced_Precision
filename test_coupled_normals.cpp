// test_coupled_normals.cpp

#include "mlmc_rng.cpp"  //or could change and call AVX512 header files directly
#include "algorithm"

void debug_coupling_extremes(int num_samples) {
    printf("Debugging coupling for extreme cases:\n");
    
    for (int i = 0; i < num_samples; ++i) {
        float u = uni01(rng_eng);
        _Float16 u16 = (_Float16)u;
        

        float z32_single, z16_single;
        lut_piecewise_constant_fp32_avx512(1, &u, &z32_single);   //change to dyadic/superdyadic based on tests
        
        _Float16 u16_arr[1] = {u16};
        _Float16 z16_arr[1];
        lut_piecewise_constant_fp16_avx512(1, &u, z16_arr, lut_fp16_dup);    //change to dyadic/superdyadic based on tests
        float z16_float = (float)z16_arr[0];
        
        float diff = z32_single - z16_float;
        
        if (fabs(diff) > 1e-3 || u < 0.001 || u > 0.999) {
            printf("u=%.10f, u16=%.10f, z32=%.8f, z16=%.8f, diff=%.8f",
                   u, (float)u16, z32_single, z16_float, diff);
            
            if (fabs(diff) > 1e-3) printf("  *** LARGE DIFF ***");
            printf("\n");
        }
    }
}

void analyse_coupling_distribution(int num_samples) {
    std::vector<float> differences;
    std::vector<float> extreme_uniforms;
    int total_large_diffs = 0;
    
    double sum_diff = 0.0;
    double sum_diff_sq = 0.0;
    float max_diff = 0.0;
    
    for (int i = 0; i < num_samples; ++i) {
        float u = uni01(rng_eng);
        
        float z32, z16_float;
        lut_piecewise_constant_fp32_avx512(1, &u, &z32);
        

        _Float16 z16_arr[1];
        lut_piecewise_constant_fp16_avx512(1, &u, z16_arr, lut_fp16_dup);
        z16_float = (float)z16_arr[0];
        
        float diff = z32 - z16_float;
        differences.push_back(diff);
        
        sum_diff += diff;
        sum_diff_sq += diff * diff;
        max_diff = std::max(max_diff, std::abs(diff));
        
        if (std::abs(diff) > 1e-3) {
            total_large_diffs++;
            extreme_uniforms.push_back(u);
        }
    }
    
    double mean_diff = sum_diff / num_samples;
    double mean_sq_diff = sum_diff_sq / num_samples;
    double stddev_diff = std::sqrt(mean_sq_diff - mean_diff * mean_diff);
    
    printf("Coupling analysis (%d samples):\n", num_samples);
    printf("Mean difference:    %.6e\n", mean_diff);
    printf("Max difference:     %.6e\n", max_diff);
    printf("Variance:  %.6e\n", stddev_diff*stddev_diff);
    printf("Large diffs (>1e-3): %d (%.3f%%)\n", 
           total_large_diffs, 100.0 * total_large_diffs / num_samples);
    
    if (!extreme_uniforms.empty()) {
        std::sort(extreme_uniforms.begin(), extreme_uniforms.end());
        printf("Extreme differences occur when u in [%.6f, %.6f]\n",
               extreme_uniforms.front(), extreme_uniforms.back());
        
        int near_zero = 0, near_one = 0, middle = 0;
        for (float u : extreme_uniforms) {
            if (u < 0.001) near_zero++;
            else if (u > 0.999) near_one++;
            else middle++;
        }
        printf("Extreme cases: near_zero=%d, near_one=%d, middle=%d\n",
               near_zero, near_one, middle);
    }
    
    std::sort(differences.begin(), differences.end(), 
              [](float a, float b) { return std::abs(a) < std::abs(b); });
    
    printf("Difference percentiles (absolute):\n");
    printf("50%%: %.3e, 90%%: %.3e, 95%%: %.3e, 99%%: %.3e, 99.9%%: %.3e\n",
           std::abs(differences[num_samples * 0.5]),
           std::abs(differences[num_samples * 0.9]),
           std::abs(differences[num_samples * 0.95]),
           std::abs(differences[num_samples * 0.99]),
           std::abs(differences[num_samples * 0.999]));
}


void test_fixed_coupling() {
    printf("Testing fixed coupling approach:\n");
    
    for (int i = 0; i < 10000; i++) {
        float u = uni01(rng_eng);
        
        float z32;
        lut_piecewise_constant_fp32_avx512(1, &u, &z32);
        
        int idx;
        if (u < 0.5f) {
            idx = static_cast<int>(u * N_LUT * 2.0f);
        } else {
            idx = static_cast<int>((1.0f - u) * N_LUT * 2.0f);
        }
        idx = fmin(idx, N_LUT - 1);
        _Float16 z16 = GAUSS_LUT_FP16[idx];
        if (u >= 0.5f) z16 = -z16;
        
        float diff = z32 - (float)z16;
        if (fabs(diff) > 1e-6) {
            printf("Coupling error: u=%.8f, diff=%.8f\n", u, diff);
        }
    }
}

int main() {
    rng_initialisation();
    printf("\n\n");
    debug_coupling_extremes(10000000);  
    analyse_coupling_distribution(10000000);
    //test_fixed_coupling();
    printf("\n\n");
    rng_termination();
    return 0;
}

