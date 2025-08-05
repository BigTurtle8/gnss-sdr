/*!
 * \file volk_gnsssdr_32f_index_max_32u.h
 * \brief VOLK_GNSSSDR kernel: Finds and returns the index which contains the maximum value in the given vector.
 *
 *
 * -----------------------------------------------------------------------------
 *
 * GNSS-SDR is a Global Navigation Satellite System software-defined receiver.
 * This file is part of GNSS-SDR.
 *
 * Copyright (C) 2010-2020  (see AUTHORS file for a list of contributors)
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * -----------------------------------------------------------------------------
 */


/*!
 * \page volk_gnsssdr_32f_index_max_32u.h
 *
 * \b Overview
 *
 * Finds and returns the index which contains the maximum value in the given vector.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * volk_gnsssdr_32f_index_max_32u(uint32_t* target, const float* src0, uint32_t num_points)
 * \endcode
 *
 * \b Inputs
 * \li src0: The input vector of floats.
 * \li num_points: The number of data points.
 *
 * \b Outputs
 * \li target: The index of the maximum value in the input buffer.
 *
 */


#ifndef INCLUDED_volk_gnsssdr_32f_index_max_32u_H
#define INCLUDED_volk_gnsssdr_32f_index_max_32u_H

#include <volk_gnsssdr/volk_gnsssdr_common.h>
#include <inttypes.h>

#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_gnsssdr_32f_index_max_32u_a_avx(uint32_t* target, const float* src0, uint32_t num_points)
{
    if (num_points > 0)
        {
            uint32_t number = 0;
            const uint32_t quarterPoints = num_points / 8;

            float* inputPtr = (float*)src0;

            __m256 indexIncrementValues = _mm256_set1_ps(8);
            __m256 currentIndexes = _mm256_set_ps(-1, -2, -3, -4, -5, -6, -7, -8);

            float max = src0[0];
            float index = 0;
            __m256 maxValues = _mm256_set1_ps(max);
            __m256 maxValuesIndex = _mm256_setzero_ps();
            __m256 compareResults;
            __m256 currentValues;

            __VOLK_ATTR_ALIGNED(32)
            float maxValuesBuffer[8];
            __VOLK_ATTR_ALIGNED(32)
            float maxIndexesBuffer[8];

            for (; number < quarterPoints; number++)
                {
                    currentValues = _mm256_load_ps(inputPtr);
                    inputPtr += 8;
                    currentIndexes = _mm256_add_ps(currentIndexes, indexIncrementValues);
                    compareResults = _mm256_cmp_ps(maxValues, currentValues, 0x1e);
                    maxValuesIndex = _mm256_blendv_ps(currentIndexes, maxValuesIndex, compareResults);
                    maxValues = _mm256_blendv_ps(currentValues, maxValues, compareResults);
                }

            // Calculate the largest value from the remaining 8 points
            _mm256_store_ps(maxValuesBuffer, maxValues);
            _mm256_store_ps(maxIndexesBuffer, maxValuesIndex);

            for (number = 0; number < 8; number++)
                {
                    if (maxValuesBuffer[number] > max)
                        {
                            index = maxIndexesBuffer[number];
                            max = maxValuesBuffer[number];
                        }
                }

            number = quarterPoints * 8;
            for (; number < num_points; number++)
                {
                    if (src0[number] > max)
                        {
                            index = number;
                            max = src0[number];
                        }
                }
            target[0] = (uint32_t)index;
        }
}

#endif /*LV_HAVE_AVX*/


#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_gnsssdr_32f_index_max_32u_u_avx(uint32_t* target, const float* src0, uint32_t num_points)
{
    if (num_points > 0)
        {
            uint32_t number = 0;
            const uint32_t quarterPoints = num_points / 8;

            float* inputPtr = (float*)src0;

            __m256 indexIncrementValues = _mm256_set1_ps(8);
            __m256 currentIndexes = _mm256_set_ps(-1, -2, -3, -4, -5, -6, -7, -8);

            float max = src0[0];
            float index = 0;
            __m256 maxValues = _mm256_set1_ps(max);
            __m256 maxValuesIndex = _mm256_setzero_ps();
            __m256 compareResults;
            __m256 currentValues;

            __VOLK_ATTR_ALIGNED(32)
            float maxValuesBuffer[8];
            __VOLK_ATTR_ALIGNED(32)
            float maxIndexesBuffer[8];

            for (; number < quarterPoints; number++)
                {
                    currentValues = _mm256_loadu_ps(inputPtr);
                    inputPtr += 8;
                    currentIndexes = _mm256_add_ps(currentIndexes, indexIncrementValues);
                    compareResults = _mm256_cmp_ps(maxValues, currentValues, 0x1e);
                    maxValuesIndex = _mm256_blendv_ps(currentIndexes, maxValuesIndex, compareResults);
                    maxValues = _mm256_blendv_ps(currentValues, maxValues, compareResults);
                }

            // Calculate the largest value from the remaining 8 points
            _mm256_store_ps(maxValuesBuffer, maxValues);
            _mm256_store_ps(maxIndexesBuffer, maxValuesIndex);

            for (number = 0; number < 8; number++)
                {
                    if (maxValuesBuffer[number] > max)
                        {
                            index = maxIndexesBuffer[number];
                            max = maxValuesBuffer[number];
                        }
                }

            number = quarterPoints * 8;
            for (; number < num_points; number++)
                {
                    if (src0[number] > max)
                        {
                            index = number;
                            max = src0[number];
                        }
                }
            target[0] = (uint32_t)index;
        }
}

#endif /*LV_HAVE_AVX*/


#ifdef LV_HAVE_SSE4_1
#include <smmintrin.h>

static inline void volk_gnsssdr_32f_index_max_32u_a_sse4_1(uint32_t* target, const float* src0, uint32_t num_points)
{
    if (num_points > 0)
        {
            uint32_t number = 0;
            const uint32_t quarterPoints = num_points / 4;

            float* inputPtr = (float*)src0;

            __m128 indexIncrementValues = _mm_set1_ps(4);
            __m128 currentIndexes = _mm_set_ps(-1, -2, -3, -4);

            float max = src0[0];
            float index = 0;
            __m128 maxValues = _mm_set1_ps(max);
            __m128 maxValuesIndex = _mm_setzero_ps();
            __m128 compareResults;
            __m128 currentValues;

            __VOLK_ATTR_ALIGNED(16)
            float maxValuesBuffer[4];
            __VOLK_ATTR_ALIGNED(16)
            float maxIndexesBuffer[4];

            for (; number < quarterPoints; number++)
                {
                    currentValues = _mm_load_ps(inputPtr);
                    inputPtr += 4;
                    currentIndexes = _mm_add_ps(currentIndexes, indexIncrementValues);
                    compareResults = _mm_cmpgt_ps(maxValues, currentValues);
                    maxValuesIndex = _mm_blendv_ps(currentIndexes, maxValuesIndex, compareResults);
                    maxValues = _mm_blendv_ps(currentValues, maxValues, compareResults);
                }

            // Calculate the largest value from the remaining 4 points
            _mm_store_ps(maxValuesBuffer, maxValues);
            _mm_store_ps(maxIndexesBuffer, maxValuesIndex);

            for (number = 0; number < 4; number++)
                {
                    if (maxValuesBuffer[number] > max)
                        {
                            index = maxIndexesBuffer[number];
                            max = maxValuesBuffer[number];
                        }
                }

            number = quarterPoints * 4;
            for (; number < num_points; number++)
                {
                    if (src0[number] > max)
                        {
                            index = number;
                            max = src0[number];
                        }
                }
            target[0] = (uint32_t)index;
        }
}

#endif /*LV_HAVE_SSE4_1*/


#ifdef LV_HAVE_SSE4_1
#include <smmintrin.h>

static inline void volk_gnsssdr_32f_index_max_32u_u_sse4_1(uint32_t* target, const float* src0, uint32_t num_points)
{
    if (num_points > 0)
        {
            uint32_t number = 0;
            const uint32_t quarterPoints = num_points / 4;

            float* inputPtr = (float*)src0;

            __m128 indexIncrementValues = _mm_set1_ps(4);
            __m128 currentIndexes = _mm_set_ps(-1, -2, -3, -4);

            float max = src0[0];
            float index = 0;
            __m128 maxValues = _mm_set1_ps(max);
            __m128 maxValuesIndex = _mm_setzero_ps();
            __m128 compareResults;
            __m128 currentValues;

            __VOLK_ATTR_ALIGNED(16)
            float maxValuesBuffer[4];
            __VOLK_ATTR_ALIGNED(16)
            float maxIndexesBuffer[4];

            for (; number < quarterPoints; number++)
                {
                    currentValues = _mm_loadu_ps(inputPtr);
                    inputPtr += 4;
                    currentIndexes = _mm_add_ps(currentIndexes, indexIncrementValues);
                    compareResults = _mm_cmpgt_ps(maxValues, currentValues);
                    maxValuesIndex = _mm_blendv_ps(currentIndexes, maxValuesIndex, compareResults);
                    maxValues = _mm_blendv_ps(currentValues, maxValues, compareResults);
                }

            // Calculate the largest value from the remaining 4 points
            _mm_store_ps(maxValuesBuffer, maxValues);
            _mm_store_ps(maxIndexesBuffer, maxValuesIndex);

            for (number = 0; number < 4; number++)
                {
                    if (maxValuesBuffer[number] > max)
                        {
                            index = maxIndexesBuffer[number];
                            max = maxValuesBuffer[number];
                        }
                }

            number = quarterPoints * 4;
            for (; number < num_points; number++)
                {
                    if (src0[number] > max)
                        {
                            index = number;
                            max = src0[number];
                        }
                }
            target[0] = (uint32_t)index;
        }
}

#endif /*LV_HAVE_SSE4_1*/


#ifdef LV_HAVE_SSE

#include <xmmintrin.h>

static inline void volk_gnsssdr_32f_index_max_32u_a_sse(uint32_t* target, const float* src0, uint32_t num_points)
{
    if (num_points > 0)
        {
            uint32_t number = 0;
            const uint32_t quarterPoints = num_points / 4;

            float* inputPtr = (float*)src0;

            __m128 indexIncrementValues = _mm_set1_ps(4);
            __m128 currentIndexes = _mm_set_ps(-1, -2, -3, -4);

            float max = src0[0];
            float index = 0;
            __m128 maxValues = _mm_set1_ps(max);
            __m128 maxValuesIndex = _mm_setzero_ps();
            __m128 compareResults;
            __m128 currentValues;

            __VOLK_ATTR_ALIGNED(16)
            float maxValuesBuffer[4];
            __VOLK_ATTR_ALIGNED(16)
            float maxIndexesBuffer[4];

            for (; number < quarterPoints; number++)
                {
                    currentValues = _mm_load_ps(inputPtr);
                    inputPtr += 4;
                    currentIndexes = _mm_add_ps(currentIndexes, indexIncrementValues);
                    compareResults = _mm_cmpgt_ps(maxValues, currentValues);
                    maxValuesIndex = _mm_or_ps(_mm_and_ps(compareResults, maxValuesIndex), _mm_andnot_ps(compareResults, currentIndexes));
                    maxValues = _mm_or_ps(_mm_and_ps(compareResults, maxValues), _mm_andnot_ps(compareResults, currentValues));
                }

            // Calculate the largest value from the remaining 4 points
            _mm_store_ps(maxValuesBuffer, maxValues);
            _mm_store_ps(maxIndexesBuffer, maxValuesIndex);

            for (number = 0; number < 4; number++)
                {
                    if (maxValuesBuffer[number] > max)
                        {
                            index = maxIndexesBuffer[number];
                            max = maxValuesBuffer[number];
                        }
                }

            number = quarterPoints * 4;
            for (; number < num_points; number++)
                {
                    if (src0[number] > max)
                        {
                            index = number;
                            max = src0[number];
                        }
                }
            target[0] = (uint32_t)index;
        }
}

#endif /*LV_HAVE_SSE*/


#ifdef LV_HAVE_SSE

#include <xmmintrin.h>

static inline void volk_gnsssdr_32f_index_max_32u_u_sse(uint32_t* target, const float* src0, uint32_t num_points)
{
    if (num_points > 0)
        {
            uint32_t number = 0;
            const uint32_t quarterPoints = num_points / 4;

            float* inputPtr = (float*)src0;

            __m128 indexIncrementValues = _mm_set1_ps(4);
            __m128 currentIndexes = _mm_set_ps(-1, -2, -3, -4);

            float max = src0[0];
            float index = 0;
            __m128 maxValues = _mm_set1_ps(max);
            __m128 maxValuesIndex = _mm_setzero_ps();
            __m128 compareResults;
            __m128 currentValues;

            __VOLK_ATTR_ALIGNED(16)
            float maxValuesBuffer[4];
            __VOLK_ATTR_ALIGNED(16)
            float maxIndexesBuffer[4];

            for (; number < quarterPoints; number++)
                {
                    currentValues = _mm_loadu_ps(inputPtr);
                    inputPtr += 4;
                    currentIndexes = _mm_add_ps(currentIndexes, indexIncrementValues);
                    compareResults = _mm_cmpgt_ps(maxValues, currentValues);
                    maxValuesIndex = _mm_or_ps(_mm_and_ps(compareResults, maxValuesIndex), _mm_andnot_ps(compareResults, currentIndexes));
                    maxValues = _mm_or_ps(_mm_and_ps(compareResults, maxValues), _mm_andnot_ps(compareResults, currentValues));
                }

            // Calculate the largest value from the remaining 4 points
            _mm_store_ps(maxValuesBuffer, maxValues);
            _mm_store_ps(maxIndexesBuffer, maxValuesIndex);

            for (number = 0; number < 4; number++)
                {
                    if (maxValuesBuffer[number] > max)
                        {
                            index = maxIndexesBuffer[number];
                            max = maxValuesBuffer[number];
                        }
                }

            number = quarterPoints * 4;
            for (; number < num_points; number++)
                {
                    if (src0[number] > max)
                        {
                            index = number;
                            max = src0[number];
                        }
                }
            target[0] = (uint32_t)index;
        }
}

#endif /*LV_HAVE_SSE*/


#ifdef LV_HAVE_GENERIC

static inline void volk_gnsssdr_32f_index_max_32u_generic(uint32_t* target, const float* src0, uint32_t num_points)
{
    if (num_points > 0)
        {
            float max = src0[0];
            uint32_t index = 0;

            uint32_t i = 1;

            for (; i < num_points; ++i)
                {
                    if (src0[i] > max)
                        {
                            index = i;
                            max = src0[i];
                        }
                }
            target[0] = index;
        }
}

#endif /*LV_HAVE_GENERIC*/


#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void volk_gnsssdr_32f_index_max_32u_neon(uint32_t* target, const float* src0, uint32_t num_points)
{
    if (num_points > 0)
        {
            uint32_t number = 0;
            const uint32_t quarterPoints = num_points / 4;

            float* inputPtr = (float*)src0;
            float32x4_t indexIncrementValues = vdupq_n_f32(4);
            __VOLK_ATTR_ALIGNED(16)
            float currentIndexes_float[4] = {-4.0f, -3.0f, -2.0f, -1.0f};
            float32x4_t currentIndexes = vld1q_f32(currentIndexes_float);

            float max = src0[0];
            float index = 0;
            float32x4_t maxValues = vdupq_n_f32(max);
            uint32x4_t maxValuesIndex = vmovq_n_u32(0);
            uint32x4_t compareResults;
            uint32x4_t currentIndexes_u;
            float32x4_t currentValues;

            __VOLK_ATTR_ALIGNED(16)
            float maxValuesBuffer[4];
            __VOLK_ATTR_ALIGNED(16)
            float maxIndexesBuffer[4];

            for (; number < quarterPoints; number++)
                {
                    currentValues = vld1q_f32(inputPtr);
                    inputPtr += 4;
                    currentIndexes = vaddq_f32(currentIndexes, indexIncrementValues);
                    currentIndexes_u = vcvtq_u32_f32(currentIndexes);
                    compareResults = vcgtq_f32(maxValues, currentValues);
                    maxValuesIndex = vorrq_u32(vandq_u32(compareResults, maxValuesIndex), vbicq_u32(currentIndexes_u, compareResults));
                    maxValues = vmaxq_f32(currentValues, maxValues);
                }

            // Calculate the largest value from the remaining 4 points
            vst1q_f32(maxValuesBuffer, maxValues);
            vst1q_f32(maxIndexesBuffer, vcvtq_f32_u32(maxValuesIndex));
            for (number = 0; number < 4; number++)
                {
                    if (maxValuesBuffer[number] > max)
                        {
                            index = maxIndexesBuffer[number];
                            max = maxValuesBuffer[number];
                        }
                }

            number = quarterPoints * 4;
            for (; number < num_points; number++)
                {
                    if (src0[number] > max)
                        {
                            index = number;
                            max = src0[number];
                        }
                }
            target[0] = (uint32_t)index;
        }
}

#endif /* LV_HAVE_NEON */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void volk_gnsssdr_32f_index_max_32u_rvv(uint32_t* target, const float* src0, uint32_t num_points)
{
    // To make consistent with other implementations,
    // do nothing if size of buffer is 0
    if (num_points == 0)
        {
            return;
        }

    size_t n = num_points;

    // Initialize pointer of correct type
    // to keep track while strip mining
    const float* inPtr = src0;

    // max[0] = 0f
    vfloat32m1_t maxVal = __riscv_vfmv_s_f_f32m8(inPtr[0], 1);

    float prevMax = inPtr[0];

    uint32_t maxI = 0;

    // Note: See `volk_gnsssdr_8i_index_max_16u_rvv` for notes
    // on possible optimization and current limitation of hardware.
    for (size_t vl; n > 0; n -= vl, inPtr += vl * 4)
        {
            // Record number of elements that will actually be processed
            vl = __riscv_vsetvl_e32m8(n);

            // Load in[0..vl)
            vfloat32m8_t inVal = __riscv_vle32_v_f32m8(inPtr, vl);

            // max[0] = max( max[0], in[0..vl) )
            maxVal = __riscv_vfredmax_vs_f32m8_f32m1(inVal, maxVal, vl);
            const float currMax = __riscv_vfmv_f_s_f32m1_f32(maxVal);

            // If found new, larger max, find first index within that element
            if (currMax > prevMax)
                {
                    // maxTarget[i] = in[i] == max[0] ? 1 : 0
                    vbool4_t maxTargetMask = __riscv_vmfeq_vf_f32m8_b4(
                        inVal, currMax, vl
                    );

                    // maxTargetI = first set bit in maxTarget
                    uint32_t maxTargetI = (uint32_t) __riscv_vfirst_m_b4(maxTargetMask, vl);
                    // Cast is risky; keep an eye out

                    uint32_t elapsedN = num_points - n;

                    // Adjust index to correct spot in larger buffer
                    maxI = maxTargetI + elapsedN;

                    // Keep track of (now) previous sum
                    prevMax = currMax;
                }

            // On looping, decrement the number of
            // elements left and increase the pointers
            // to account for the number of elements processed
            // (taking into consideration element size)
        }

    *target = maxI;
}

#endif /*LV_HAVE_RVV*/

#endif /* INCLUDED_volk_gnsssdr_32f_index_max_32u_H */
