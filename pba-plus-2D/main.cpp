/*
Author: Cao Thanh Tung and Zheng Jiaqi
Date: 21/01/2010, 20/08/2019

File Name: main.cpp

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

#include <climits>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <random>

#include "pba/pba2D.h"

// Input parameters
int fboSize		= 26000;
int nVertices   = 100;

int phase1Band  = 32;   // should be equal or less than size / 64
int phase2Band  = 32;
int phase3Band  = 2;    // should be euqal or less than 64

// Global Variable
typedef struct {
    double totalDistError, maxDistError; 
    int errorCount; 
} ErrorStatistics;

short *inputPoints, *inputVoronoi, *outputVoronoi; 
ErrorStatistics pba; 

// Random Point Generator

// Generate input points
void generateRandomPoints(int width, int height, int nPoints)
{
    int tx, ty;

    // Initialize the Mersenne Twister RNG with a seed of 0
    // (Matching your original randinit(0) behavior)
    std::mt19937 rng(0);

    // Define uniform distributions for the exact integer ranges [0, width-1] and [0, height-1]
    std::uniform_int_distribution<int> distX(0, width - 1);
    std::uniform_int_distribution<int> distY(0, height - 1);

    // Initialize Voronoi array
    for (int i = 0; i < width * height * 2ULL; i++) {
        inputVoronoi[i] = MARKER;
    }

    // Generate random points
    for (int i = 0; i < nPoints; i++)
    {
        do {
            // Draw directly from the integer distributions
            tx = distX(rng);
            ty = distY(rng);
        } while (inputVoronoi[(ty * width + tx) * 2] != MARKER);

        // Populate arrays
        inputVoronoi[(ty * width + tx) * 2    ] = tx; 
        inputVoronoi[(ty * width + tx) * 2 + 1] = ty; 

        inputPoints[i * 2    ] = tx; 
        inputPoints[i * 2 + 1] = ty; 
    }
}

// Deinitialization
void deinitialization()
{
    pba2DDeinitialization(); 

    free(inputPoints); 
    free(inputVoronoi); 
    free(outputVoronoi); 
}

// Initialization                                                                           
#define ULL unsigned long long
void initialization()
{
    pba2DInitialization(fboSize, phase1Band); 

    inputPoints     = (short *) malloc((ULL)nVertices * 2ULL * (ULL)sizeof(short)); 
    inputVoronoi    = (short *) malloc((ULL)fboSize * (ULL)fboSize * 2ULL * (ULL)sizeof(short)); 
    outputVoronoi   = (short *) malloc((ULL)fboSize * (ULL)fboSize * 2ULL * (ULL)sizeof(short)); 
}
#undef ULL

// Verify the output Voronoi Diagram
void verifyResult(ErrorStatistics *e) 
{
    e->totalDistError = 0.0; 
    e->maxDistError = 0.0; 
    e->errorCount = 0; 

    int tx, ty; 
    double dist, myDist, correctDist, error;

    for (int i = 0; i < fboSize; i++) {
        for (int j = 0; j < fboSize; j++) {
            int id = j * fboSize + i; 

            tx = outputVoronoi[id * 2] - i; 
            ty = outputVoronoi[id * 2 + 1] - j; 
            correctDist = myDist = tx * tx + ty * ty; 

            for (int t = 0; t < nVertices; t++) {
                tx = inputPoints[t * 2] - i; 
                ty = inputPoints[t * 2 + 1] - j; 
                dist = tx * tx + ty * ty; 

                if (dist < correctDist)
                    correctDist = dist; 
            }

            if (correctDist != myDist) {
                error = fabs(sqrt(myDist) - sqrt(correctDist)); 

                e->errorCount++; 
                e->totalDistError += error; 

                if (error > e->maxDistError)
                    e->maxDistError = error; 
            }
        }
    }
}

void printStatistics(ErrorStatistics *e)
{
    double avgDistError = e->totalDistError / e->errorCount; 

    if (e->errorCount == 0)
        avgDistError = 0.0; 

    printf("* Error count           : %i -> %.3f\n", e->errorCount, 
        (double(e->errorCount) / nVertices) * 100.0);
    printf("* Max distance error    : %.5f\n", e->maxDistError);
    printf("* Average distance error: %.5f\n", avgDistError);
}

// Run the tests
void runTests()
{
    generateRandomPoints(fboSize, fboSize, nVertices); 

    pba2DVoronoiDiagram(inputVoronoi, outputVoronoi, phase1Band, phase2Band, phase3Band); 

    printf("Verifying the result...\n"); 
    // verifyResult(&pba);

    printf("-----------------\n");
    printf("Texture: %dx%d\n", fboSize, fboSize);
    printf("Points: %d\n", nVertices);
    printf("-----------------\n");

    printStatistics(&pba); 
}

int main(int argc,char **argv)
{
    initialization();

    runTests();

    deinitialization();

	return 0;
}