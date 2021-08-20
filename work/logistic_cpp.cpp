#include <iostream>
#include <math.h>

using namespace std;

using T = double;

constexpr T range_min = 2.0;
constexpr T range_max = 4.0;
constexpr auto range = range_max - range_min;

constexpr auto all_threads = 1024;
constexpr auto block_size = 16;
constexpr auto thread_size = all_threads / block_size;
constexpr auto warmup = 300;
constexpr auto loops = 1000;

T get_a(int blockId, int threadId) {
    return // 2 * log2(2 + range / thread_size * threadId);
        range_min +
        // ((T)thread_size) / ((T)threadIdx.x) * range
        range / (thread_size * block_size) * (blockId * thread_size + threadId);
}

void logistic() {
    for(int blockId = 0; blockId < block_size; blockId++) {
        for(int threadId = 0; threadId < thread_size; threadId++) {
            const auto a = get_a(blockId, threadId);
            T x = 0.8;
            for(int index = 0; index < warmup; index++) {
                x = a * x * (1 - x);
            }
            for(int index = warmup; index < loops; index++) {
                printf("%1.8lf %1.8lf\n", a, x);
                x = a * x * (1 - x);
            }
        }
    }
}

int main() { logistic(); }