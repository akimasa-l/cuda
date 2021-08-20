#include <iostream>

using namespace std;

using T = float;

constexpr T range_min = 2.0;
constexpr T range_max = 4.0;
constexpr T range = range_max - range_min;

constexpr auto all_threads = 1024 * 4;
constexpr auto block_size = 8;
constexpr auto thread_size = all_threads / block_size;
constexpr auto warmup = 300;
constexpr auto loops = 1000;

__device__ T get_a() {
    return // 2 * log2(2 + range / thread_size * threadIdx.x);
        range_min +
        // ((T)thread_size) / ((T)threadIdx.x) * range
        range / (thread_size * block_size) *
            (blockIdx.x * thread_size + threadIdx.x);
}

__global__ void logistic() {
    const auto a = get_a();
    T x = 0.8;
    for(int index = 0; index < warmup; index++) {
        x = a * x * (1 - x);
    }
    for(int index = warmup; index < loops; index++) {
        printf("%1.8lf %1.8lf\n", a, x);
        x = a * x * (1 - x);
    }
}

int main() {
    // size_t nBytes = loops * (thread_size + 1) * sizeof(T);
    // T *x;
    // cudaMalloc((void **)&x, nBytes);
    logistic<<<block_size, thread_size>>>();
    // cudaFree(x);
    cudaDeviceReset();
}
