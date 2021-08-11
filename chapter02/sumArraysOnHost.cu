#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/*
 * This example demonstrates a simple vector sum on the host. sumArraysOnHost
 * sequentially iterates through vector elements on the host.
 */

__global__ void showResultOnDevice(float *d_A, float *d_B, float *d_C) {

    printf("d_A is %g\n", d_A[0]);
    printf("d_B is %g\n", d_B[0]);
    printf("d_C is %g\n", d_C[0]);
}

__global__ void sumArraysOnDevice(float *A, float *B,
                                  float *C /* , const int N */) {
    C[threadIdx.x] = A[threadIdx.x] + B[threadIdx.x];
}

void sumArraysOnHost(float *A, float *B, float *C, const int N) {
    for(int idx = 0; idx < N; idx++) {
        C[idx] = A[idx] + B[idx];
    }
}

void initialData(float *ip, int size) {
    // generate different seed for random number
    time_t t;
    srand((unsigned)time(&t));

    for(int i = 0; i < size; i++) {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }

    return;
}

int main(int argc, char **argv) {
    const int nElem = 1024;
    const size_t nBytes = nElem * sizeof(float);

    float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;

    h_A = (float *)malloc(nBytes); // alloc memory on cpu
    h_B = (float *)malloc(nBytes);
    h_C = (float *)malloc(nBytes);

    cudaMalloc((float **)&d_A, nBytes); // alloc memory on gpu
    cudaMalloc((float **)&d_B, nBytes);
    cudaMalloc((float **)&d_C, nBytes);


    initialData(h_A, nElem);
    initialData(h_B, nElem);

    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice); // copy value to gpu
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

    dim3 block(nElem);
    dim3 grid(1);

    sumArraysOnHost(h_A, h_B, h_C, nElem);             // calc on cpu
    sumArraysOnDevice<<<grid, block>>>(d_A, d_B, d_C); // calc on gpu

    printf("h_A is %g\n", h_A[0]);
    printf("h_B is %g\n", h_B[0]);
    printf("h_C is %g\n", h_C[0]);

    showResultOnDevice<<<1, 1>>>(d_A, d_B, d_C);
    
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaDeviceReset();
    return (0);
}
