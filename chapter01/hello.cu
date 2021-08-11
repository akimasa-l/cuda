#include <stdio.h>
__global__ void helloFromGPU() {
    const auto a = threadIdx.x;
    printf("Hello World From GPU thread %d!\n", a);
}
int main() {
    printf("Hello World From CPU1!\n");
    helloFromGPU<<<1, 10>>>();
    printf("Hello World From CPU2!\n");
    cudaDeviceReset();
    // cudaDeviceSynchronize();
    printf("Hello World From CPU3!\n");
    return 0;
}