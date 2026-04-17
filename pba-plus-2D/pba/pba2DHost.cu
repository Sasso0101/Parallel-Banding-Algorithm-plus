/*
Author: Cao Thanh Tung and Zheng Jiaqi
Date: 21/01/2010, 20/08/2019

File Name: pba2DHost.cu

===============================================================================

Copyright (c) 2019, School of Computing, National University of Singapore. 
All rights reserved.

Project homepage: http://www.comp.nus.edu.sg/~tants/pba.html

If you use PBA and you like it or have comments on its usefulness etc., we 
would love to hear from you at <tants@comp.nus.edu.sg>. You may share with us
your experience and any possibilities that we may improve the work/code.

===============================================================================

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/

#include <cuda_runtime.h>
#include <iostream>

#include "pba2D.h"

// Macro adapted for standard CUDA runtime error checking
#define CUDA_CHECK(status) \
    do { \
        cudaError_t _status = (status); \
        if (_status != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(_status) \
                      << " at line " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Parameters for CUDA kernel executions
#define BLOCKSIZE   64

// Global Variables
short2 **pbaTextures;       // Two textures used to compute 2D Voronoi Diagram
short2 *margin;

size_t pbaMemSize;          // Size (in bytes) of a texture
int pbaTexSize;             // Texture size (squared texture)

// Kernels
#include "pba2DKernel.h"

// Initialize necessary memory for 2D Voronoi Diagram computation
// - textureSize: The size of the Discrete Voronoi Diagram (width = height)
// - phase1Band: number of bands of phase 1
#define ULL unsigned long long
void pba2DInitialization(int textureSize, int phase1Band)
{
    pbaTexSize = textureSize;

    pbaMemSize = (ULL)pbaTexSize * (ULL)pbaTexSize * (ULL)sizeof(short2);

    pbaTextures = (short2 **) malloc(2 * sizeof(short2 *));

    std::cout << "Total memory allocated: " << 2 * pbaMemSize + 2ULL * (ULL)phase1Band * (ULL)pbaTexSize * sizeof(short2) << std::endl;

    CUDA_CHECK(cudaMalloc((void **) &pbaTextures[0], pbaMemSize));

    CUDA_CHECK(cudaMalloc((void **) &pbaTextures[1], pbaMemSize));

    CUDA_CHECK(cudaMalloc((void **) &margin, 2ULL * (ULL)phase1Band * (ULL)pbaTexSize * sizeof(short2)));

}
#undef ULL

// Deallocate all allocated memory
void pba2DDeinitialization()
{
    CUDA_CHECK(cudaFree(pbaTextures[0]));
    CUDA_CHECK(cudaFree(pbaTextures[1]));
    CUDA_CHECK(cudaFree(margin));

    free(pbaTextures);
}

// Copy input to GPU
void pba2DInitializeInput(short *input)
{
    CUDA_CHECK(cudaMemcpy(pbaTextures[0], input, pbaMemSize, cudaMemcpyHostToDevice));
}

// Phase 1 of PBA. m1 must divides texture size and equal or less than size / 64
void pba2DPhase1(int m1)
{
    dim3 grid = dim3(pbaTexSize / BLOCKSIZE, m1);

    kernelFloodDown<<< grid, BLOCKSIZE >>>(pbaTextures[0], pbaTextures[0], pbaTexSize, pbaTexSize / m1);
    CUDA_CHECK(cudaGetLastError());

    kernelFloodUp<<< grid, BLOCKSIZE >>>(pbaTextures[0], pbaTextures[0], pbaTexSize, pbaTexSize / m1);
    CUDA_CHECK(cudaGetLastError());

    kernelPropagateInterband<<< grid, BLOCKSIZE >>>(pbaTextures[0], margin, pbaTexSize, pbaTexSize / m1);
    CUDA_CHECK(cudaGetLastError());

    kernelUpdateVertical<<< grid, BLOCKSIZE >>>(pbaTextures[0], margin, pbaTextures[1], pbaTexSize, pbaTexSize / m1);
    CUDA_CHECK(cudaGetLastError());
}

// Phase 2 of PBA. m2 must divides texture size
void pba2DPhase2(int m2)
{
    // Compute proximate points locally in each band
    dim3 grid = dim3(pbaTexSize / BLOCKSIZE, m2);

    kernelProximatePoints<<< grid, BLOCKSIZE >>>(pbaTextures[1], pbaTextures[0], pbaTexSize, pbaTexSize / m2);
    CUDA_CHECK(cudaGetLastError());

    kernelCreateForwardPointers<<< grid, BLOCKSIZE >>>(pbaTextures[0], pbaTextures[0], pbaTexSize, pbaTexSize / m2);
    CUDA_CHECK(cudaGetLastError());

    // Repeatly merging two bands into one
    for (int noBand = m2; noBand > 1; noBand /= 2) {
        grid = dim3(pbaTexSize / BLOCKSIZE, noBand / 2);
        kernelMergeBands<<< grid, BLOCKSIZE >>>(pbaTextures[1], pbaTextures[0], pbaTextures[0], pbaTexSize, pbaTexSize / noBand);
        CUDA_CHECK(cudaGetLastError());
    }

    // Replace the forward link with the X coordinate of the seed to remove
    // the need of looking at the other texture. We need it for coloring.
    grid = dim3(pbaTexSize / BLOCKSIZE, pbaTexSize);
    kernelDoubleToSingleList<<< grid, BLOCKSIZE >>>(pbaTextures[1], pbaTextures[0], pbaTextures[0], pbaTexSize);
    CUDA_CHECK(cudaGetLastError());
}

// Phase 3 of PBA. m3 must divides texture size and equal or less than 64
void pba2DPhase3(int m3)
{
    dim3 block = dim3(BLOCKSIZE, m3);
    dim3 grid = dim3(pbaTexSize / block.x);

    kernelColor<<< grid, block >>>(pbaTextures[0], pbaTextures[1], pbaTexSize);
    CUDA_CHECK(cudaGetLastError());
}

void pba2DCompute(int m1, int m2, int m3)
{
    pba2DPhase1(m1);

    pba2DPhase2(m2);

    pba2DPhase3(m3);
}

// Compute 2D Voronoi diagram
// Input: a 2D texture. Each pixel is represented as two "short" integer.
//    For each site at (x, y), the pixel at coordinate (x, y) should contain
//    the pair (x, y). Pixels that are not sites should contain the pair (MARKER, MARKER)
// See original paper for the effect of the three parameters: m1, m2, m3
// Parameters must divide textureSize
void pba2DVoronoiDiagram(short *input, short *output, int m1, int m2, int m3)
{
    // Initialization
    pba2DInitializeInput(input);

    // --- Create CUDA Events for Timing ---
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Record start event
    CUDA_CHECK(cudaEventRecord(start));

    // Computation
    pba2DCompute(m1, m2, m3);

    // Record stop event and wait for GPU to finish
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    // Calculate and print elapsed time
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "PBA Kernel Execution Time: " << milliseconds << " ms\n";

    // Clean up events
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    // -------------------------------------

    // Copy back the result
    CUDA_CHECK(cudaMemcpy(output, pbaTextures[1], pbaMemSize, cudaMemcpyDeviceToHost));
}