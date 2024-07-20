#include <stdio.h>
#include <cuda_runtime.h>

__global__ void matrixAddKernel(float* A, float* B, float* C, int N){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int index = i * N + j;
    if(i < N && j < N){
        A[index] = B[index] + C[index];
    }
}


void matrixAdd(float* A, float* B, float* C, int N){
    int size = N * N * sizeof(float);
    float *d_A, *d_B, *d_C;

    cudaMalloc((void **)&d_A, size);

    cudaMalloc((void **)&d_B, size);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_C, size);
    cudaMemcpy(d_C, C, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                        (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrixAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    cudaMemcpy(A, d_A, size, cudaMemcpyDeviceToHost);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}

int main(){
    const int n = 2;
    float a[n * n];
    float b[n * n] = {1, 2, 3, 4};
    float c[n * n] = {5, 6, 7, 8};

    matrixAdd(a, b, c, n);

    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            printf("%1.0f ", a[i * n + j]);
        }
        printf("\n");
    }
    cudaDeviceSynchronize();
    return 0;
}