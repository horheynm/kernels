#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#ifndef TILE_DIM
#define TILE_DIM 16
#endif

// Input is MxN (row-major). Output is NxM (row-major) and equals input^T.
__global__ void transpose_naive(const float* input, float* output, int M, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x; // 0..N-1
    int row = blockIdx.y * blockDim.y + threadIdx.y; // 0..M-1
    if (row < M && col < N) {
        output[col * M + row] = input[row * N + col];
    }
}

static void transpose_cpu(const float* input, float* output, int M, int N) {
    for (int r = 0; r < M; ++r) {
        for (int c = 0; c < N; ++c) {
            output[c * M + r] = input[r * N + c];
        }
    }
}

int main(int argc, char** argv) {
    int M = 32, N = 32;

    size_t bytes = (size_t)M * (size_t)N * sizeof(float);

    std::vector<float> h_in((size_t)M * (size_t)N);
    std::vector<float> h_out((size_t)M * (size_t)N, 0.0f);
    std::vector<float> h_ref((size_t)M * (size_t)N, 0.0f);

    for (int r = 0; r < M; ++r) {
        for (int c = 0; c < N; ++c) {
            h_in[(size_t)r * (size_t)N + (size_t)c] = (float)(r * 1000 + c);
        }
    }

    float* d_in = nullptr;
    float* d_out = nullptr;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, bytes);

    dim3 threadsPerBlock(TILE_DIM, TILE_DIM);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    transpose_naive<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, M, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost);

    transpose_cpu(h_in.data(), h_ref.data(), M, N);

    for (size_t i = 0; i < h_ref.size(); ++i) {
        if (h_out[i] != h_ref[i]) {
            std::fprintf(stderr,
                         "FAIL: mismatch at linear idx=%zu: got=%f expected=%f\n",
                         i, h_out[i], h_ref[i]);
            cudaFree(d_in);
            cudaFree(d_out);
            return 1;
        }
    }

    std::printf("PASS: transpose correct for M=%d N=%d\n", M, N);

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}