#include <thrust/complex.h>
#include <complex>

using F = double;
using T = thrust::complex<F>;

constexpr F range_x_max = +1;
constexpr F range_x_min = -1;
constexpr F range_y_max = +1;
constexpr F range_y_min = -1;
constexpr int block_x = 64;
constexpr int block_y = 64;
constexpr int thread_x = 16;
constexpr int thread_y = 16;

inline T get_place(const int ix, const int iy) {
    return {range_x_min + (range_x_max - range_x_min) * (F)ix /
                              ((F)block_x * (F)thread_x),
            range_y_min + (range_y_max - range_y_min) * (F)iy /
                              ((F)block_y * (F)thread_y)};
}

inline int get_id(const int ix, const int iy) {
    return block_x * thread_x * ix + iy;
}

inline T newton_method(T x) {
    const T f = 1 / (1 + thrust::exp(x));
    const T df = f * (1 - f);
    return df.real() == .0 && df.imag() == .0 ? df : x - f / df;
}

void calc(const int ix, const int iy, F *h_x) {
    const auto id = get_id(ix, iy);
    auto c = get_place(ix, iy);
    for(int i = 0; i < 100; i++) {
        c = newton_method(c);
    }
    h_x[id] = thrust::arg(c);
}

void print(F *h_x) {
    for(int ix = 0; ix < block_x * thread_x; ix++) {
        printf("%lg", h_x[block_x * thread_x * ix]);
        for(int iy = 1; iy < block_y * thread_y; iy++) {
            const auto id = get_id(ix, iy);
            calc(ix, iy, h_x);
            printf(",%lg", h_x[id]);
        }
        printf("\n");
    }
}

int main() {
    size_t size = block_x * block_y * thread_x * thread_y * sizeof(F);
    F *h_x;
    h_x = (F *)malloc(size);
    print(h_x);
    free(h_x);
}