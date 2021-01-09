// Include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#include <x86intrin.h>
#endif

// Include OpenMP
#include <omp.h>

#include "mandelbrot.h"
#include "parameters.h"

uint32_t iterations(struct parameters params, double complex point) {
    double complex z = 0;
    for (int i = 1; i <= params.maxiters; i++) {
        z = z * z + point;
        if (creal(z) * creal(z) + cimag(z) * cimag(z) >= params.threshold * params.threshold) {
            return i;
        }
    }
    return 0;
}

void mandelbrot(struct parameters params, double scale, int32_t *num_pixels_in_set) {
    int32_t num_zero_pixels = 0;
    __m256d thresh = _mm256_set1_pd(params.threshold * params.threshold);
    __m256d mask = _mm256_set1_pd(1);
    double cImag = cimag(params.center);
    double cReal = creal(params.center);
    #pragma omp parallel for reduction(+ : num_zero_pixels) 
    for (int i = params.resolution; i >= -params.resolution; i--) {
        __m256d yCord = _mm256_set1_pd(cImag + i * scale / params.resolution);
        for (int j = 0; j < 2 * params.resolution / 4 * 4; j += 4) {
            double xCord0 = cReal + (j - params.resolution) * scale / params.resolution;
            double xCord1 = cReal + (j + 1 - params.resolution) * scale / params.resolution;
            double xCord2 = cReal + (j + 2 - params.resolution) * scale / params.resolution;
            double xCord3 = cReal + (j + 3 - params.resolution) * scale / params.resolution;
            __m256d xCord = _mm256_set_pd(xCord0, xCord1, xCord2, xCord3);
    	    __m256d zImag = _mm256_setzero_pd();
    	    __m256d zReal = _mm256_setzero_pd();
	    double s[4];
   	       for (int i = 1; i <= params.maxiters; i ++) {
                //zImag = (ZReal * ZImag * 2) + yCord
        	   //zReal = (zReal * zReal) - (zImag * zImag) + xCord
        	   __m256d zIR = _mm256_mul_pd(zImag, zReal);
        	   __m256d zRsq = _mm256_mul_pd(zReal, zReal);
        	   __m256d zIsq = _mm256_mul_pd(zImag, zImag);
        	   zImag = _mm256_add_pd(_mm256_add_pd(zIR, zIR), yCord);
        	   zReal = _mm256_add_pd(_mm256_sub_pd(zRsq, zIsq), xCord);
        	   //comp = zImag * zImag + zReal * zReal
                   __m256d comp = _mm256_add_pd(_mm256_mul_pd(zImag, zImag), _mm256_mul_pd(zReal, zReal));
        	   comp = _mm256_and_pd(_mm256_cmp_pd(comp, thresh, 1), mask);
        	   //store comp into array
        	   _mm256_storeu_pd(s, comp);
        	   if (s[0] == 0 && s[1] == 0 && s[2] == 0 && s[3] == 0) {
           		   break;
        	   }
    	    }
            num_zero_pixels += (uint32_t) s[0] + s[1] + s[2] + s[3];         
        }
        for(int t = 2 * params.resolution / 4 * 4; t <= 2 * params.resolution; t++) {
            double complex tailPoint = (params.center +
                    (t - params.resolution) * scale / params.resolution +
                    i * scale / params.resolution * I);
            if (iterations(params, tailPoint) == 0) {
                num_zero_pixels ++;
            }
        }
    }
    *num_pixels_in_set = num_zero_pixels;
}

