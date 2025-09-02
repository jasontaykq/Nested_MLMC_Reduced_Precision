//fp16x32.h
#pragma once

#include <immintrin.h>
#include <iostream>

// SIMD wrapper for 32 half-precision floats using AVX-512 FP16

class fp16x32 {
public:
    __m512h data;

    fp16x32() = default;
    explicit fp16x32(__m512h val) : data(val) {}
    explicit fp16x32(_Float16 val) {
        data = _mm512_set1_ph(val);
    }
    explicit fp16x32(float val) {
        data = _mm512_set1_ph((_Float16)val);
    }

    fp16x32 operator+(const fp16x32& other) const {
        return fp16x32(_mm512_add_ph(data, other.data));
    }
    fp16x32 operator-(const fp16x32& other) const {
        return fp16x32(_mm512_sub_ph(data, other.data));
    }
    fp16x32 operator*(const fp16x32& other) const {
        return fp16x32(_mm512_mul_ph(data, other.data));
    }
    fp16x32 operator/(const fp16x32& other) const {
        return fp16x32(_mm512_div_ph(data, other.data));
    }

    fp16x32& operator+=(const fp16x32& other) {
        data = _mm512_add_ph(data, other.data);
        return *this;
    }
    fp16x32& operator-=(const fp16x32& other) {
        data = _mm512_sub_ph(data, other.data);
        return *this;
    }
    fp16x32& operator*=(const fp16x32& other) {
        data = _mm512_mul_ph(data, other.data);
        return *this;
    }
    fp16x32& operator/=(const fp16x32& other) {
        data = _mm512_div_ph(data, other.data);
        return *this;
    }
};


// === Free Functions ===

// Fused ops:  +/- (a*b) +/- c

inline fp16x32 fmadd(const fp16x32& a, const fp16x32& b, const fp16x32& c) {
    return fp16x32(_mm512_fmadd_ph(a.data, b.data, c.data));
}

inline fp16x32 fmsub(const fp16x32& a, const fp16x32& b, const fp16x32& c) {
    return fp16x32(_mm512_fmsub_ph(a.data, b.data, c.data));
}

inline fp16x32 fnmadd(const fp16x32& a, const fp16x32& b, const fp16x32& c) {
    return fp16x32(_mm512_fnmadd_ph(a.data, b.data, c.data));
}

inline fp16x32 fnmsub(const fp16x32& a, const fp16x32& b, const fp16x32& c) {
    return fp16x32(_mm512_fnmsub_ph(a.data, b.data, c.data));
}

// Math ops using SVML + FP16-safe casting

inline fp16x32 exp(const fp16x32& x) {
    __m256h lo = _mm512_castph512_ph256(x.data);
    __m256h hi = _mm256_castsi256_ph(_mm512_extracti64x4_epi64(
				     _mm512_castph_si512(x.data), 1));
    __m512 lo32 = _mm512_cvtph_ps(lo), hi32 = _mm512_cvtph_ps(hi);
    __m512 lo_exp = _mm512_exp_ps(lo32), hi_exp = _mm512_exp_ps(hi32);
    __m256h lo_res = _mm512_cvtps_ph(lo_exp, _MM_FROUND_TO_NEAREST_INT |
				             _MM_FROUND_NO_EXC);
    __m256h hi_res = _mm512_cvtps_ph(hi_exp, _MM_FROUND_TO_NEAREST_INT |
				             _MM_FROUND_NO_EXC);
    __m512h result = _mm512_castph256_ph512(lo_res);
    return fp16x32(_mm512_castsi512_ph(_mm512_inserti64x4(
                                       _mm512_castph_si512(result),
                                       _mm256_castph_si256(hi_res), 1)));
}

inline fp16x32 log(const fp16x32& x) {
    __m256h lo = _mm512_castph512_ph256(x.data);
    __m256h hi = _mm256_castsi256_ph(_mm512_extracti64x4_epi64(
				     _mm512_castph_si512(x.data), 1));
    __m512 lo32 = _mm512_cvtph_ps(lo),   hi32 = _mm512_cvtph_ps(hi);
    __m512 lo_log = _mm512_log_ps(lo32), hi_log = _mm512_log_ps(hi32);
    __m256h lo_res = _mm512_cvtps_ph(lo_log, _MM_FROUND_TO_NEAREST_INT |
				             _MM_FROUND_NO_EXC);
    __m256h hi_res = _mm512_cvtps_ph(hi_log, _MM_FROUND_TO_NEAREST_INT |
				             _MM_FROUND_NO_EXC);
    __m512h result = _mm512_castph256_ph512(lo_res);
    return fp16x32(_mm512_castsi512_ph(_mm512_inserti64x4(
                                       _mm512_castph_si512(result),
                                       _mm256_castph_si256(hi_res), 1)));
}

// reciprocal, square root and reciprocal square root functions

inline fp16x32 rcp(const fp16x32& x) {
    return fp16x32(_mm512_rcp_ph(x.data));
}

inline fp16x32 sqrt(const fp16x32& x) {
    return fp16x32(_mm512_sqrt_ph(x.data));
}

inline fp16x32 rsqrt(const fp16x32& x) {
    return fp16x32(_mm512_rsqrt_ph(x.data));
}


// Load/store operations, with and without alignment

inline fp16x32 aligned_load(const _Float16* ptr) {
    return fp16x32(_mm512_load_ph(ptr));
}

inline fp16x32 unaligned_load(const _Float16* ptr) {
    return fp16x32(_mm512_loadu_ph(ptr));
}

inline void aligned_store(_Float16* ptr, const fp16x32& x) {
    _mm512_store_ph(ptr, x.data);
}

inline void unaligned_store(_Float16* ptr, const fp16x32& x) {
    _mm512_storeu_ph(ptr, x.data);
}

// Debug print function

inline void debug_print(const char* label, const fp16x32& x) {
    alignas(64) _Float16 tmp[32];
    _mm512_store_ph(tmp, x.data);
    std::cout << label << "[";
    for (int i = 0; i < 32; ++i) {
        std::cout << static_cast<float>(tmp[i]) << (i != 31 ? ", " : "");
    }
    std::cout << "]\n";
}

// 16-bit gather functions, assuming data is provided as an array
// of 32-bit values each duplicating the 16-bit value; this is due
// to the lack of an AVX gather instruction for 16-bit data

inline __m512i gather_epi16(const int* base, __m512i indices16) {
    __m512i even_lanes = _mm512_set_epi16(
            30,30,28,28,26,26,24,24,
            22,22,20,20,18,18,16,16,
            14,14,12,12,10,10,8,8,
             6, 6, 4, 4, 2, 2, 0, 0
        );

    __m512i odd_lanes = _mm512_set_epi16(
            31,31,29,29,27,27,25,25,
            23,23,21,21,19,19,17,17,
            15,15,13,13,11,11,9,9,
             7, 7, 5, 5, 3, 3, 1, 1
        );

    __m512i index_even = _mm512_maskz_permutexvar_epi16(0x55555555,
                                             even_lanes, indices16);
    __m512i index_odd  = _mm512_maskz_permutexvar_epi16(0x55555555,
                                              odd_lanes, indices16);

    __m512i gathered_even = _mm512_i32gather_epi32(index_even, base, 4);
    __m512i gathered_odd  = _mm512_i32gather_epi32(index_odd,  base, 4);

    return _mm512_mask_blend_epi16(0xAAAAAAAA,gathered_even,gathered_odd);
}

inline fp16x32 gather_fp16(const _Float16* base, __m512i indices16) {
    return fp16x32(_mm512_castsi512_ph(gather_epi16((int*) base,indices16)));
}
