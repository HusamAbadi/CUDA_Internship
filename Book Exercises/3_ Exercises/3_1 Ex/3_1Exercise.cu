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


// Stub Function
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
    const int n = 10;
    float a[n * n];
    float b[n * n] = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
        31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
        41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
        51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
        61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
        71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
        81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
        91, 92, 93, 94, 95, 96, 97, 98, 99, 100
    };
    
    float c[n * n] = {
        100, 99, 98, 97, 96, 95, 94, 93, 92, 91,
        90, 89, 88, 87, 86, 85, 84, 83, 82, 81,
        80, 79, 78, 77, 76, 75, 74, 73, 72, 71,
        70, 69, 68, 67, 66, 65, 64, 63, 62, 61,
        60, 59, 58, 57, 56, 55, 54, 53, 52, 51,
        50, 49, 48, 47, 46, 45, 44, 43, 42, 41,
        40, 39, 38, 37, 36, 35, 34, 33, 32, 31,
        30, 29, 28, 27, 26, 25, 24, 23, 22, 21,
        20, 19, 18, 17, 16, 15, 14, 13, 12, 11,
        10, 9, 8, 7, 6, 5, 4, 3, 2, 1
    };

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