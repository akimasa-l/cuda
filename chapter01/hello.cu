#include <stdio.h>
__global__ void helloFromGPU() {
    const auto a = threadIdx.x;
    printf("Hello World From GPU thread %d!\n", a);
}
int main() {
    printf("Hello World From CPU!\n");
    helloFromGPU<<<1, 10>>>();
    printf("Hello World From CPU!\n");
    // cudaDeviceReset();
    cudaDeviceSynchronize();
    return 0;
}