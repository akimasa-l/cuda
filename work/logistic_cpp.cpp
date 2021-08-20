#include <iostream>
#include <math.h>

using namespace std;

using T = double;

constexpr T range_min = 2.0;
constexpr T range_max = 4.0;
constexpr auto range = range_max - range_min;

constexpr auto thread_size = 256;
constexpr auto warmup = 300;
constexpr auto loops = 400;

T get_a(int threadId) {
    return 2 * log2(2 + range / thread_size * threadId);
    /* range_min +
// ((T)thread_size) / ((T)threadIdx.x) * range
           range / thread_size * threadIdx.x; */
}

void init(T *x) {}

void logistic(T *x) {
    for(int threadId = 0; threadId < thread_size; threadId++) {
        const auto a = get_a(threadId);
        x[threadId] = 0.8;
        for(int index = 0; index < warmup; index++) {
            const auto new_index = threadId + thread_size * index;
            x[new_index + thread_size] = a * x[new_index] * (1 - x[new_index]);
        }
        for(int index = warmup; index < loops; index++) {
            const auto new_index = threadId + thread_size * index;
            printf("%1.8lf %1.8lf\n", a, x[new_index]);
            x[new_index + thread_size] = a * x[new_index] * (1 - x[new_index]);
        }
    }
}

int main() {
    size_t nBytes = loops * (thread_size + 1) * sizeof(T);
    T *x;
    x = (T *)malloc(nBytes);
    logistic(x);
    free(x);
}