#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include "image.h"

#define RUNS 200

#define TBW 32 // ThreadBlock width
#define TBH  8 // ThreadBlock height
#define TBWE (TBW+2)
#define TBHE (TBH+2)

__constant__ float xFilterSobel[3][3] = {
    {-1.0f,  0.0f,  1.0f},
    {-2.0f,  0.0f,  2.0f},
    {-1.0f,  0.0f,  1.0f}
};

__constant__ float yFilterSobel[3][3] = {
    {-1.0f, -2.0f, -1.0f},
    { 0.0f,  0.0f,  0.0f},
    { 1.0f,  2.0f,  1.0f}
};

__global__ void filterSobelCuda(unsigned char* imageIn, unsigned char* imageOut, int width)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int id  = threadIdx.y * blockDim.x + threadIdx.x;
    int loadAddress = (blockIdx.y*blockDim.y)*(width+2) + blockIdx.x*blockDim.x;
    float xValue=0.0f, yValue=0.0f;

    __shared__ unsigned char window[TBHE][TBWE];
    if(id < TBWE*TBHE/2) {
        window[(id/TBWE)       ][id%TBWE] = imageIn[loadAddress + ((id/TBWE)       )*(width+2) + id%TBWE];
        window[(id/TBWE)+TBHE/2][id%TBWE] = imageIn[loadAddress + ((id/TBWE)+TBHE/2)*(width+2) + id%TBWE];
    }
    __syncthreads();

    #pragma unroll 3
    for(int mrow=0; mrow<3; ++mrow) {
        #pragma unroll 3
        for(int mcol=0; mcol<3; ++mcol) {
            xValue += (float)(window[threadIdx.y + mrow][threadIdx.x + mcol]) * xFilterSobel[mrow][mcol];
            yValue += (float)(window[threadIdx.y + mrow][threadIdx.x + mcol]) * yFilterSobel[mrow][mcol];
        }
    }
    imageOut[row*width + col] = (unsigned char)fmaxf(fminf(sqrtf(xValue*xValue + yValue*yValue),255.0f),0.0f);
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        printf("Usage: %s <file_in> <file_out>\n", argv[0]);
        exit(1);
    }

    t_image   imageIn,   imageOut;
    t_image_f gsImageIn;

    if(readPngFile(argv[1], &imageIn)) {
        printf("Error while reading image\n");
        exit(1);
    }
    rgb2gs(imageIn, &gsImageIn);

    imageOut.width    = imageIn.width;
    imageOut.height   = imageIn.height;
    mallocImage(&imageOut);
    freeImage(imageIn);

    unsigned char *gsImageInF = (unsigned char*)malloc(sizeof(unsigned char) * gsImageIn.width * gsImageIn.height);
    for(int y=0; y<gsImageIn.height; ++y) {
        for(int x=0; x<gsImageIn.width; ++x) {
            gsImageInF[y*gsImageIn.width + x] = (unsigned char)gsImageIn.data[y][x];
        }
    }

    unsigned char *kernelIn;
    unsigned char *kernelOut;
    cudaMalloc((void**)&kernelIn,  gsImageIn.width * gsImageIn.height * sizeof(unsigned char));
    cudaMalloc((void**)&kernelOut, imageOut.width  * imageOut.height  * sizeof(unsigned char));
    cudaMemcpy(kernelIn, gsImageInF, gsImageIn.width * gsImageIn.height * sizeof(*kernelIn), cudaMemcpyHostToDevice);
    freeImageF(gsImageIn);

    dim3 thrBlock(TBW, TBH);
    dim3 thrGrid(imageOut.width/TBW, imageOut.height/TBH);

    printf("Times for %d iterations:\n", RUNS);
    cudaDeviceSynchronize();
    clock_t start = clock();

    for(int i=0; i<RUNS; ++i) {
        filterSobelCuda<<<thrGrid, thrBlock>>>(kernelIn, kernelOut, imageOut.width);
    }

    cudaDeviceSynchronize();
    clock_t end = clock();
    float runTime = (float)(end - start) / CLOCKS_PER_SEC; // Runtime in seconds
    float numberOfMegaPixels = (imageOut.width * imageOut.height) / (1000.0f * 1000.0f);
    printf("%8.3f MegaPixels/sec (%8.3f ms, %8.3f fps)\n",
            numberOfMegaPixels*RUNS/runTime,
            runTime*1000,
            RUNS/runTime
       );

    unsigned char *gsImageOut = (unsigned char*)malloc(sizeof(unsigned char) * imageOut.width  * imageOut.height);
    cudaMemcpy(gsImageOut, kernelOut, imageOut.width * imageOut.height * sizeof(*kernelOut), cudaMemcpyDeviceToHost);
    for(int y=0; y<imageOut.height; ++y) {
        for(int x=0; x<imageOut.width; ++x) {
            imageOut.data[y][x].r = gsImageOut[y*imageOut.width + x];
            imageOut.data[y][x].g = gsImageOut[y*imageOut.width + x];
            imageOut.data[y][x].b = gsImageOut[y*imageOut.width + x];
        }
    }

    if(writePngFile(argv[2], imageOut)) {
        printf("Error while saving image\n");
        exit(1);
    }

    cudaFree(kernelIn);
    cudaFree(kernelOut);
    freeImage(imageOut);
    free(gsImageOut);
    free(gsImageInF);

    return 0;
}
