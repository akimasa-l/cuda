#include <thrust/complex.h>
#include <tuple>

using F = double;
using T = thrust::complex<F>;

constexpr F range_x_max = +2;
constexpr F range_x_min = -2;
constexpr F range_y_max = +2;
constexpr F range_y_min = -2;
constexpr int block_x = 256;
constexpr int block_y = 256;
constexpr int thread_x = 32;
constexpr int thread_y = 32;

__device__ inline int get_ix() { return threadIdx.x + blockIdx.x * blockDim.x; }
__device__ inline int get_iy() { return threadIdx.y + blockIdx.y * blockDim.y; }
__device__ inline int get_id(const int ix, const int iy) {
    return block_x * thread_x * ix + iy;
}

__device__ inline T get_place(const int ix, const int iy) {
    return {range_x_min + (range_x_max - range_x_min) * (F)ix /
                              ((F)block_x * (F)thread_x),
            range_y_min + (range_y_max - range_y_min) * (F)iy /
                              ((F)block_y * (F)thread_y)};
}

__device__ inline T newton_method(T x) {
    const T f = 1 / (1 + thrust::exp(-2 * x));
    const T df =
        2 * thrust::exp(-2 * x) / thrust::pow(thrust::exp(-2 * x) + 1, 2);
    return df.real() == .0 && df.imag() == .0 ? df : x - f / df;
}

__global__ void calc(F *d_x) {

    const auto ix = get_ix(), iy = get_iy();
    const auto id = get_id(ix, iy);
    auto c = get_place(ix, iy);
    /* printf("%lg %lg\n", c.real(), c.imag());
    c = newton_method(c);
    printf("%lg %lg\n", c.real(), c.imag());
    printf("%lg\n",thrust::arg(c)); */
    for(int i = 0; i < 100; i++) {
        c = newton_method(c);
    }
    d_x[id] = thrust::arg(c);
    // printf("%lg\n",d_x[id]);
    // printf("%lg\n", thrust::arg(c));
    // printf("id : %d\n", get_id(get_ix(), get_iy()));
    // printf("place is %lg %lg\n", place.real(), place.imag());
}

void print(F *h_x) {
    for(int x = 0; x < block_x * thread_x; x++) {
        printf("%lg", h_x[block_x * thread_x * x]);
        for(int y = 1; y < block_y * thread_y; y++) {
            printf(",%lg", h_x[block_x * thread_x * x + y]);
        }
        printf("\n");
    }
}

int main() {
    size_t size = block_x * block_y * thread_x * thread_y * sizeof(F);
    F *h_x, *d_x;
    cudaMalloc((void **)&d_x, size);
    dim3 block(block_x, block_y);
    dim3 thread(thread_x, thread_y);
    calc<<<block, thread>>>(d_x);
    h_x = (F *)malloc(size);
    cudaMemcpy(h_x, d_x, size, cudaMemcpyDeviceToHost);
    print(h_x);
    cudaFree(d_x);
    free(h_x);
    cudaDeviceReset();
}