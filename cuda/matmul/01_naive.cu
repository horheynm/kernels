#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

__global__ void gemm(
    const float* A, // [M, K]
    const float* B, // [K, N]
    float* C,       // [M, N]
    int M,
    int N,
    int K
) {
    /* 
        A = [M, K], B = [K, N], C = [M, N]
    */
    const int col = threadIdx.x + blockDim.x * blockIdx.x;
    const int row = threadIdx.y + blockDim.y * blockIdx.y;


    if (row < M && col < N) {
        
        float acc = 0.f;
        for (int k=0; k < K; ++k) {
            acc += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = acc;
    }
}

static void cpu_gemm(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.f;
            for (int k = 0; k < K; ++k) {
                acc += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = acc;
        }
    }
}

int main(int argc, char** argv) {
    int M = 512; // rows of A, C
    int N = 512; // cols of B, C
    int K = 256; // cols of A, rows of B

    if (argc == 4) {
        M = std::atoi(argv[1]);
        N = std::atoi(argv[2]);
        K = std::atoi(argv[3]);
    } else if (argc != 1) {
        printf("Usage: %s [M N K]\n", argv[0]);
        return 1;
    }
    if (M <= 0 || N <= 0 || K <= 0) {
        printf("M, N, K must be > 0\n");
        return 1;
    }

    size_t bytesA = (size_t)M * K * sizeof(float);
    size_t bytesB = (size_t)K * N * sizeof(float);
    size_t bytesC = (size_t)M * N * sizeof(float);

    float* h_A = (float*)malloc(bytesA);
    float* h_B = (float*)malloc(bytesB);
    float* h_C = (float*)malloc(bytesC);
    float* h_C_ref = (float*)malloc(bytesC);

    if (!h_A || !h_B || !h_C || !h_C_ref) {
        printf("Host malloc failed\n");
        return 1;
    }

    for (int i = 0; i < M * K; ++i) h_A[i] = (float)(i % 13) * 0.1f;
    for (int i = 0; i < K * N; ++i) h_B[i] = (float)(i % 17) * 0.05f;

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    cudaMalloc((void**)&d_A, bytesA);
    cudaMalloc((void**)&d_B, bytesB);
    cudaMalloc((void**)&d_C, bytesC);

    cudaMemcpy(d_A, h_A, bytesA, cudaMemcpyHostToDevice);   
    cudaMemcpy(d_B, h_B, bytesB, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid(
        (N + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (M + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    gemm<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, bytesC, cudaMemcpyDeviceToHost);

    cpu_gemm(h_A, h_B, h_C_ref, M, N, K);

    int success = 1;
    for (int row = 0; row < M && success; ++row) {
        for (int col = 0; col < N; ++col) {
            float ref = h_C_ref[row * N + col];
            float got = h_C[row * N + col];
            float diff = fabsf(got - ref);
            if (diff > 1e-2f) {
                printf("Error at (row=%d, col=%d): got %f, expected %f (diff=%f)\n",
                       row, col, got, ref, diff);
                success = 0;
                break;
            }
        }
    }

    if (success) {
        printf("All checked elements are correct. M=%d N=%d K=%d\n", M, N, K);
    }

    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return success ? 0 : 1;
}
