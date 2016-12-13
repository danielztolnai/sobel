#define _XOPEN_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <stdint.h>
#include <immintrin.h>
#include "image.h"

#define MEASURE_TIME
#define RUNS 200

static const float float255 = 255.0f;
static __m256 constfloat255;
static __m256 constfloat0;

static float xFilterSobel[3][3] = {
    {-1.0f,  0.0f,  1.0f},
    {-2.0f,  0.0f,  2.0f},
    {-1.0f,  0.0f,  1.0f}
};

static float yFilterSobel[3][3] = {
    {-1.0f, -2.0f, -1.0f},
    { 0.0f,  0.0f,  0.0f},
    { 1.0f,  2.0f,  1.0f}
};

float min(float a, float b)
{
    if(a<b) {
        return a;
    } else {
        return b;
    }
}

float max(float a, float b)
{
    if(a>b) {
        return a;
    } else {
        return b;
    }
}

void filterSobelNaive(t_image_f imageIn, t_image imageOut)
{
    float xValue, yValue;
    unsigned char gsValue;

    int x, y, kx, ky;

    for (y=1; y<imageIn.height-1; ++y) {
        for (x=1; x<imageIn.width-1; ++x) {
            xValue=0.0f;
            yValue=0.0f;
            for(ky=0; ky<3; ++ky) {
                for(kx=0; kx<3; ++kx) {
                    xValue += imageIn.data[y-1+ky][x-1+kx] * xFilterSobel[ky][kx];
                    yValue += imageIn.data[y-1+ky][x-1+kx] * yFilterSobel[ky][kx];
                }
            }
            gsValue = (unsigned char)max(min(floor(sqrt(xValue*xValue + yValue*yValue)),255),0);
            imageOut.data[y-1][x-1].r = gsValue;
            imageOut.data[y-1][x-1].g = gsValue;
            imageOut.data[y-1][x-1].b = gsValue;
        }
    }
}

void filterSobelThreaded(t_image_f imageIn, t_image imageOut)
{
    float xValue, yValue;
    unsigned char gsValue;

    int x, y, kx, ky;

    #pragma omp parallel for private(xValue, yValue, x, kx, ky, gsValue) shared(imageIn, imageOut)
    for (y=1; y<imageIn.height-1; ++y) {
        for (x=1; x<imageIn.width-1; ++x) {
            xValue=0.0f;
            yValue=0.0f;
            for(ky=0; ky<3; ++ky) {
                for(kx=0; kx<3; ++kx) {
                    xValue += imageIn.data[y-1+ky][x-1+kx] * xFilterSobel[ky][kx];
                    yValue += imageIn.data[y-1+ky][x-1+kx] * yFilterSobel[ky][kx];
                }
            }
            gsValue = (unsigned char)max(min(floor(sqrt(xValue*xValue + yValue*yValue)),255),0);
            imageOut.data[y-1][x-1].r = gsValue;
            imageOut.data[y-1][x-1].g = gsValue;
            imageOut.data[y-1][x-1].b = gsValue;
        }
    }
    #pragma omp barrier
}

void filterSobelVector(t_image_f imageIn, t_image imageOut)
{
    __m256 pixelValue, filterValue;
    __m256 xValue, yValue;
    __m256i gsValueVector;
    int gsValue[8];

    int x, y, kx, ky;

    #pragma omp parallel for private(xValue, yValue, x, kx, ky, gsValue, gsValueVector, filterValue, pixelValue) shared(imageIn, imageOut)
    for (y=1; y<imageIn.height-1; ++y) {
        for (x=1; x<imageIn.width-1; x+=8) {
            xValue = _mm256_setzero_ps();
            yValue = _mm256_setzero_ps();
            for(ky=0; ky<3; ++ky) {
                for(kx=0; kx<3; ++kx) {
                    pixelValue  = _mm256_loadu_ps(&(imageIn.data[y-1+ky][x-1+kx]));
                    filterValue = _mm256_broadcast_ss(&(xFilterSobel[ky][kx]));
                    xValue = _mm256_add_ps (xValue, _mm256_mul_ps (pixelValue, filterValue));
                    filterValue = _mm256_broadcast_ss(&(yFilterSobel[ky][kx]));
                    yValue = _mm256_add_ps (yValue, _mm256_mul_ps (pixelValue, filterValue));
                }
            }
            xValue = _mm256_mul_ps(xValue, xValue);
            yValue = _mm256_mul_ps(yValue, yValue);
            xValue = _mm256_add_ps(xValue, yValue);
            xValue = _mm256_sqrt_ps(xValue);
            xValue = _mm256_max_ps(xValue, constfloat0);
            xValue = _mm256_min_ps(xValue, constfloat255);
            gsValueVector = _mm256_cvtps_epi32(xValue);
            _mm256_maskstore_epi32 (gsValue, _mm256_set1_epi8(255), gsValueVector);
            for(kx=0; kx<8; ++kx) {
                imageOut.data[y-1][x-1+kx].r = (unsigned char)gsValue[kx];
                imageOut.data[y-1][x-1+kx].g = (unsigned char)gsValue[kx];
                imageOut.data[y-1][x-1+kx].b = (unsigned char)gsValue[kx];
            }
        }
    }
    #pragma omp barrier
}

const char* getFunctionName(void (*functionPtr)(t_image_f,t_image))
{
    if(functionPtr == &filterSobelNaive)
        return "filterSobelNaive";
    if(functionPtr == &filterSobelThreaded)
        return "filterSobelThreaded";
    if(functionPtr == &filterSobelVector)
        return "filterSobelVector";
    return "invalid";
}

void benchmarkFunction(t_image_f imageIn, t_image imageOut, void (*functionPtr)(t_image_f,t_image))
{
    double start = omp_get_wtime();

    for(int i=0; i<RUNS; ++i) {
        (*functionPtr)(imageIn, imageOut);
    }

    double end = omp_get_wtime();
    float runTime = (float)(end - start) * 1000;
    printf("%s:\t %8.3f ms (%8.3f fps)\n", getFunctionName(functionPtr), runTime, RUNS/runTime*1000);
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        printf("Usage: %s <file_in> <file_out>\n", argv[0]);
        exit(1);
    }

    constfloat255 = _mm256_broadcast_ss(&float255);
    constfloat0   = _mm256_setzero_ps();

    t_image   imageIn,   imageOut;
    t_image_f gsImageIn;

    if(readPngFile(argv[1], &imageIn)) {
        printf("Error while reading image\n");
        freeImage(imageIn);
        exit(1);
    }
    rgb2gs(imageIn, &gsImageIn);

    imageOut.width    = imageIn.width;
    imageOut.height   = imageIn.height;
    mallocImage(&imageOut);

    putenv("OMP_PROC_BIND=true");
    omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(7); // Use 7 threads for all consecutive parallel regions

    printf("Times for %d iterations:\n", RUNS);

    benchmarkFunction(gsImageIn, imageOut, &filterSobelNaive);
    benchmarkFunction(gsImageIn, imageOut, &filterSobelThreaded);
    benchmarkFunction(gsImageIn, imageOut, &filterSobelVector);

    if(writePngFile(argv[2], imageOut)) {
        printf("Error while saving image\n");
        freeImage(imageIn);
        freeImage(imageOut);
        exit(1);
    }

    freeImage(imageIn);
    freeImage(imageOut);
    freeImageF(gsImageIn);

    return 0;
}
