#include <stdio.h>

void vecAdd(float* A, float* B, float* C, int n){
    int size = n * sizeof(float);
    float *d_A, *d_B, *d_C;

    printf("Hi!");

    cudaMalloc((void **) &d_A, size);
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    
    cudaMalloc((void **) &d_B, size);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    
    cudaMalloc((void **) &d_C, size);
    
    //Kernel Invocation code - to be shown later

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

}


int main(){
    const int N = 5;
    float a[5] = {1, 2, 3, 4, 5};
    float b[5] = {5, 4, 3, 2, 1};
    float c[5];

    vecAdd(a, b, c, N);
}