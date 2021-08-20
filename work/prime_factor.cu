#include <iostream>

__global__ void fac() { printf("aa\n"); }

int main() { fac<<<1, 10>>>(); }