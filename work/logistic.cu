#include <iostream>

using namespace std;

using T = double;

constexpr T range_min = 2.0;
constexpr T range_max = 4.0;
constexpr auto range = range_max - range_min;

constexpr auto thread_size = 1024;
constexpr auto warmup = 500;
constexpr auto loops = 1000;

__device__ T get_a() {
    return 2 * log2(2 + range / thread_size * threadIdx.x)
        /* range_min +
    // ((T)thread_size) / ((T)threadIdx.x) * range
               range / thread_size * threadIdx.x; */
        ;
}

__global__ void init(T *x) {}

__global__ void logistic(T *x) {
    const auto a = get_a();
    x[threadIdx.x] = 0.8;
    for(int index = 0; index < warmup; index++) {
        const auto new_index = threadIdx.x + thread_size * index;
        x[new_index + thread_size] = a * x[new_index] * (1 - x[new_index]);
    }
    for(int index = warmup; index < loops; index++) {
        const auto new_index = threadIdx.x + thread_size * index;
        printf("%1.8lf %1.8lf\n", a, x[new_index]);
        x[new_index + thread_size] = a * x[new_index] * (1 - x[new_index]);
    }
}

int main() {
    size_t nBytes = loops * (thread_size + 1) * sizeof(T);
    T *x;
    cudaMalloc((void **)&x, nBytes);
    logistic<<<1, thread_size>>>(x);
    cudaFree(x);
    cudaDeviceReset();
}