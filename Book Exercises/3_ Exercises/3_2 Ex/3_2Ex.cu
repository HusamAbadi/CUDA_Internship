#include <stdio.h>
#include <cuda_runtime.h>

__global__ void matrixVecMultKernel(float* A, float* B, float* C, int N) {
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    if (row < N) {
        float result = 0.0f;
        for (int col = 0; col < N; ++col) {
            result += B[row * N + col] * C[col];
        }
        A[row] = result;
    }
}


void matrixVecMult(float* A, float* B, float* C, int N){
    int sizeA = N * sizeof(float);         // Size for the result vector
    int sizeB = N * N * sizeof(float);     // Size for the matrix
    int sizeC = N * sizeof(float);         // Size for the input vector
    float *d_A, *d_B, *d_C;

    cudaMalloc((void **)&d_A, sizeA);
    cudaMalloc((void **)&d_B, sizeB);
    cudaMalloc((void **)&d_C, sizeC);

    cudaMemcpy(d_B, B, sizeB, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, sizeC, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x);

    matrixVecMultKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    cudaMemcpy(A, d_A, sizeA, cudaMemcpyDeviceToHost);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}

int main(){
    const int n = 2;
    float a[n]; // Result vector
    float b[n * n] = {1, 2, 3, 4}; // 2x2 matrix
    float c[n] = {5, 6}; // Vector

    matrixVecMult(a, b, c, n);
    // Print the result vector
    for (int i = 0; i < n; i++) {
        printf("%1.0f ", a[i]);
    }
    printf("\n");

    cudaDeviceSynchronize();
    return 0;
}