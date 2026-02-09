

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

__global__ void FusedKernel(
    const float* input,
    float* output,
    float alpha,
    float beta,
    int n
) {
    const float4* input4 = reinterpret_cast<const float4*>(input);
    float4* output4 = reinterpret_cast<float4*>(output);

    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;
    int n4 = n / 4;

    for (int i=idx; i < n4; i += stride){ 
        float4 in4 = input4[i];
        float4 out4 = output4[i];

        out4.x = fmaxf(0.f, in4.x * alpha + beta);
        out4.y = fmaxf(0.f, in4.y * alpha + beta);
        out4.z = fmaxf(0.f, in4.z * alpha + beta);
        out4.w = fmaxf(0.f, in4.w * alpha + beta);

        output4[i] = out4;
    }

    // handle tail
    for (int i = n4 * 4 + idx; i < n; i+=stride) {
        output[i] = fmaxf(0.f, input[i] * alpha + beta);
    }

}

static void cpu_fused(const float* input, float* output, float alpha, float beta, int n) {
    for (int i = 0; i < n; ++i) {
        float value = input[i] * alpha + beta;
        output[i] = (value > 0.0f) ? value : 0.0f;
    }
}

int main(int argc, char** argv) {
    int n = 1 << 20; // default: 1M elements
    if (argc == 2) {
        n = std::atoi(argv[1]);
    } else if (argc != 1) {
        std::printf("Usage: %s [n]\n", argv[0]);
        return 1;
    }
    if (n <= 0) {
        std::printf("n must be > 0\n");
        return 1;
    }

    const float alpha = 0.5f;
    const float beta = 0.2f;

    size_t bytes = (size_t)n * sizeof(float);
    float* h_in = (float*)std::malloc(bytes);
    float* h_out = (float*)std::malloc(bytes);
    float* h_ref = (float*)std::malloc(bytes);
    if (!h_in || !h_out || !h_ref) {
        std::printf("Host malloc failed\n");
        return 1;
    }

    for (int i = 0; i < n; ++i) {
        // mix of negative/positive values
        h_in[i] = (float)((i % 200) - 100) * 0.01f;
    }

    float *d_in = nullptr, *d_out = nullptr;
    cudaError_t err = cudaSuccess;

    err = cudaMalloc((void**)&d_in, bytes);
    if (err != cudaSuccess) {
        std::printf("cudaMalloc(d_in) failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    err = cudaMalloc((void**)&d_out, bytes);
    if (err != cudaSuccess) {
        std::printf("cudaMalloc(d_out) failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::printf("cudaMemcpy H2D failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    int threads = 256;
    dim3 threadsPerBlock(threads);
    dim3 blocksPerGrid((n + threads - 1) / threads);

    FusedKernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, alpha, beta, n);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::printf("Kernel execution failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::printf("cudaMemcpy D2H failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cpu_fused(h_in, h_ref, alpha, beta, n);

    int ok = 1;
    for (int i = 0; i < n; ++i) {
        float diff = std::fabs(h_out[i] - h_ref[i]);
        if (diff > 1e-6f) {
            std::printf("Mismatch at i=%d: got=%f ref=%f diff=%f\n", i, h_out[i], h_ref[i], diff);
            ok = 0;
            break;
        }
    }

    std::printf("%s. n=%d alpha=%f beta=%f\n", ok ? "OK" : "FAIL", n, alpha, beta);

    cudaFree(d_in);
    cudaFree(d_out);
    std::free(h_in);
    std::free(h_out);
    std::free(h_ref);
    return ok ? 0 : 1;
}