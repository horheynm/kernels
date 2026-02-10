#define TILE_DIM   16
#define BLOCK_ROWS 8

#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

__global__ void transpose(const float* input, float* output, int M, int N) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1]; // avoids bank conflicts

    int x = blockIdx.x * TILE_DIM + threadIdx.x; // 0..15
    int y = blockIdx.y * TILE_DIM + threadIdx.y; // 0..7

    // load global to shared
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        int yy = y + j;
        if (x < N && yy < M) {
            tile[threadIdx.y + j][threadIdx.x] = input[yy * N + x];
        }
    }

    __syncthreads();

    int ox = blockIdx.y * TILE_DIM + threadIdx.x; // col in output (N x M)
    int oy = blockIdx.x * TILE_DIM + threadIdx.y; // row in output

    // store shared to global
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        int oyy = oy + j;
        if (ox < M && oyy < N) {
            output[oyy * M + ox] = tile[threadIdx.x][threadIdx.y + j];
        }
    }

    
}

static void transpose_cpu(const float* input, float* output, int M, int N) {
    for (int r = 0; r < M; ++r) {
        for (int c = 0; c < N; ++c) {
            output[c * M + r] = input[r * N + c];
        }
    }
}

int main() {
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

    dim3 threadsPerBlock(TILE_DIM, BLOCK_ROWS);
    dim3 blocksPerGrid(
        (N + TILE_DIM - 1) / TILE_DIM,
        (M + TILE_DIM - 1) / TILE_DIM
    );

    transpose<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, M, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost);

    transpose_cpu(h_in.data(), h_ref.data(), M, N);
    for (size_t i = 0; i < h_ref.size(); ++i) {
        if (h_out[i] != h_ref[i]) {
            std::printf("FAIL: mismatch at linear idx=%zu: got=%f expected=%f\n",
                        i, h_out[i], h_ref[i]);
            cudaFree(d_in);
            cudaFree(d_out);
            return 1;
        }
    }
    std::printf("PASS: transpose correct for M=%d N=%d\n", M, N);

    cudaFree(d_in);
    cudaFree(d_out);
}
