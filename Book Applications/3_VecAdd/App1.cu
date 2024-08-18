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
    
    //Kernel Invocation code
    vecAddKernel<<<ceil(n/256.0), 256>>>(d_A, d_B, d_C, n);

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);


}


int main(){
    const int n = 5;
    float a[n] = {1, 2, 3, 4, 5};
    float b[n] = {10, 11, 12, 13, 14};
    float c[n];

    // Calling the Stub Function
    vecAdd(a, b, c, n);

    for(int i = 0; i < n; i++){
        printf("%1.0f ", c[i]);
    }
    cudaDeviceSynchronize();

}