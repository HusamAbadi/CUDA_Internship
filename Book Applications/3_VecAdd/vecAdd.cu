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
    cudaMalloc((void **) &d_B, size);
    cudaMalloc((void **) &d_C, size);
    
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    
    
    //Setup our kernel launch parameters
    int threads = 32;
    int blocks = (int)ceil(n / threads);

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
    const int n = 1 << 5;
    // n = 32

    float a[n] = {47, 30, 14, 70, 21, 97, 65, 78, 61, 86, 2, 20, 88, 45, 80, 33, 94, 54, 24, 50, 68, 36, 83, 57, 79, 11, 7, 72, 69, 89, 40, 16};

    float b[n] = {23, 67, 91, 13, 25, 72, 89, 3, 98, 61, 11, 85, 31, 69, 45, 16, 87, 53, 44, 6, 96, 5, 29, 82, 59, 10, 93, 2, 95, 81, 33, 64};

        
    float c[n];

    // Calling the Stub Function
    vecAdd(a, b, c, n);

    for(int i = 0; i < n; i++){
        printf("%1.0f ", c[i]);
    }
    printf("\n");

    cudaDeviceSynchronize();

}