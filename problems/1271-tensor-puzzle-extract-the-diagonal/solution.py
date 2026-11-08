#include <cuda_runtime.h>
#include <vector>

__global__ void diag_kernel(const float* A, float* out, int n) {
    // out[i] = A[i*n + i]
}

std::vector<float> diag(const std::vector<std::vector<float>>& A) {
    return {};
}
