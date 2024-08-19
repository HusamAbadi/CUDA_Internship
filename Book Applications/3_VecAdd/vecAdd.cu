#include <stdio.h>

__global__
void vecAddKernel(float* A, float* B, float* C, int n){
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if(i<n) C[i] = A[i] + B[i];
}

// Stub Function
void vecAdd(float* A, float* B, float* C, int n){
    int size = n * sizeof(float);
    float *d_A, *d_B, *d_C;

    cudaMalloc((void **) &d_A, size);
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    
    cudaMalloc((void **) &d_B, size);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    
    cudaMalloc((void **) &d_C, size);
    
    //Setup our kernel launch parameters
    int threads = 128;
    int blocks = (n + threads - 1) / threads;

    //Launch Kernel
    vecAddKernel<<<blocks, threads>>>(d_A, d_B, d_C, n);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);


}


int main(){
    const int n = 1 << 7;

    float a[n] = {47, 30, 14, 70, 21, 97, 65, 78, 61, 86, 2, 20, 88, 45, 80, 33, 94, 54, 24, 50, 68, 36, 83, 57, 79, 11, 7, 72, 69, 89, 40, 16,
    17, 10, 49, 95, 31, 75, 23, 98, 91, 41, 90, 52, 84, 48, 34, 55, 66, 18, 60, 1, 74, 59, 81, 4, 56, 92, 28, 76, 19, 64, 5, 38,
    26, 43, 96, 42, 67, 9, 85, 12, 25, 15, 99, 63, 58, 29, 77, 27, 3, 6, 73, 22, 39, 13, 71, 87, 53, 32, 8, 35, 44, 0, 51, 62, 
    46, 100, 82, 6, 9, 4, 37, 14, 55, 8, 12, 93, 27, 95, 86, 57, 35, 60, 78, 49, 61, 65, 99, 22, 68, 76, 79, 30, 7};

    float b[n] = {23, 67, 91, 13, 25, 72, 89, 3, 98, 61, 11, 85, 31, 69, 45, 16, 87, 53, 44, 6, 96, 5, 29, 82, 59, 10, 93, 2, 95, 81, 33, 64,
        18, 94, 19, 77, 40, 9, 80, 54, 50, 34, 74, 58, 0, 32, 21, 48, 55, 38, 4, 62, 49, 35, 92, 47, 43, 1, 28, 7, 37, 75, 90, 36,
        85, 99, 78, 26, 9, 95, 6, 51, 88, 8, 70, 20, 46, 77, 52, 68, 57, 92, 65, 41, 42, 15, 24, 66, 30, 83, 62, 50, 64, 63, 81, 100,
        73, 84, 38, 14, 56, 3, 59, 60, 88, 58, 42, 98, 16, 66, 13, 1, 27, 25, 31, 97, 34, 19, 48, 99, 15, 20, 17, 36, 75, 39};

        
    float c[n];

    // Calling the Stub Function
    vecAdd(a, b, c, n);

    for(int i = 0; i < n; i++){
        printf("%1.0f ", c[i]);
    }
    printf("\n");

    cudaDeviceSynchronize();

}