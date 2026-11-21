#pragma once
#include "tiled-kernel.h"

#include <cstring>

inline void zen_pack_group(const tiled_tile_src1 & source, unsigned char * packed, int first_col) {
    for (int k = 0; k < 256; k += 4) {
        for (int col = 0; col < 16; ++col) {
            std::memcpy(packed + (col / 8 * 64 + k / 4) * 32 + (col % 8) * 4, &source.q[(first_col + col) * 256 + k],
                        4);
        }
    }
}

inline __m256i zen_broadcast_four(const uint8_t * weights) {
    int bits;
    std::memcpy(&bits, weights, sizeof(bits));
    return _mm256_set1_epi32(bits);
}

inline void zen_tile(const tiled_tile_src0 & weights,
                     const tiled_tile_src1 & act,
                     const unsigned char *   packed,
                     int                     i0,
                     int                     j0,
                     float *                 buf,
                     int                     stride) {
    for (int group = 0; group < 2; ++group) {
        const int col = j0 + group * 8;
        for (int i = i0; i < i0 + 16; i += 2) {
            __m256i dots[2] = {}, mins[2] = {};
            for (int s = 0; s < 8; ++s) {
                const __m256i bs0   = _mm256_load_si256((const __m256i *) &act.bsums[(2 * s) * 256 + col]);
                const __m256i bs1   = _mm256_load_si256((const __m256i *) &act.bsums[(2 * s + 1) * 256 + col]);
                const __m256i bsum  = _mm256_add_epi32(bs0, bs1);
                const auto *  block = (const __m256i *) (packed + group * 2048 + s * 256);
#pragma GCC unroll 2
                for (int r = 0; r < 2; ++r) {
                    const auto *  w     = weights.q + (i + r) * 256 + s * 32;
                    const __m256i scale = _mm256_set1_epi16(weights.scales[(i + r) * 8 + s]);
                    mins[r]             = _mm256_add_epi32(
                        mins[r], _mm256_mullo_epi32(bsum, _mm256_set1_epi32(weights.mins[(i + r) * 8 + s])));
// Four maddubs sums fit signed i16 for both Q4_K and Q5_K: 4*2*31*128 = 31744.
#pragma GCC unroll 2
                    for (int half = 0; half < 2; ++half) {
                        const int     k     = half * 4;
                        const __m256i p0    = _mm256_maddubs_epi16(zen_broadcast_four(w + 4 * k), block[k]);
                        const __m256i p1    = _mm256_maddubs_epi16(zen_broadcast_four(w + 4 * k + 4), block[k + 1]);
                        const __m256i p2    = _mm256_maddubs_epi16(zen_broadcast_four(w + 4 * k + 8), block[k + 2]);
                        const __m256i p3    = _mm256_maddubs_epi16(zen_broadcast_four(w + 4 * k + 12), block[k + 3]);
                        const __m256i pairs = _mm256_add_epi16(_mm256_add_epi16(p0, p1), _mm256_add_epi16(p2, p3));
                        dots[r]             = _mm256_add_epi32(dots[r], _mm256_madd_epi16(pairs, scale));
                    }
                }
            }
#pragma GCC unroll 2
            for (int r = 0; r < 2; ++r) {
                __m256 res  = _mm256_mul_ps(_mm256_set1_ps(weights.d[i + r]), _mm256_cvtepi32_ps(dots[r]));
                res         = _mm256_fnmadd_ps(_mm256_set1_ps(weights.dmin[i + r]), _mm256_cvtepi32_ps(mins[r]), res);
                float * out = buf + (i + r) * stride + col;
                _mm256_storeu_ps(out, _mm256_fmadd_ps(res, _mm256_loadu_ps(&act.d[col]), _mm256_loadu_ps(out)));
            }
        }
    }
}
