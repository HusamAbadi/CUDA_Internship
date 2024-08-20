// ! Using cudaMalloc()

#include <stdio.h>
#include <cstdlib>
#include <cassert>
#include <iostream>
#include <vector>
#include <algorithm>

__global__
void vecAddKernel(int* A, int* B, int* C, int n){
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if(i<n) C[i] = A[i] + B[i];
}

int main(){
    constexpr int N = 1 << 9;
    size_t bytes = N * sizeof(int);

    // Vectors for holding the host-side data
    std::vector<int> a(N);
    std::vector<int> b(N);
    std::vector<int> c(N);
    
    // Initializing the vectors with random numbers in each array between 0-100
    std::generate(begin(a), end(a), []() { return rand() % 100; });
    std::generate(begin(b), end(b), []() { return rand() % 100; });
    
    // Allocate memory on the GPU
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    
    // Copy data to the GPU
    cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);
    
    // Thread per CTA (256 Threads)
    int threads = 1 << 8;
    // Blocks per Grid (1 Block)
    int blocks = (int)ceil(N / (float)threads);
    
    // Setup our kernel launch parameters
    dim3 block(threads, 1, 1);
    dim3 grid(blocks, 1, 1);
    
    // Launch our kernel
    vecAddKernel<<<grid, block>>>(d_a, d_b, d_c, N);
    
    // Copy data back to the host
    cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    // Display results on the terminal
    for(int i = 0; i < N; i++){
        std::cout << "a[" << i << "] + b[" << i << "] = " << a[i] << " + " << b[i] << " = " << c[i] << std::endl;
    }

    // Free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    std::cout << "Success!" << std::endl;

    return 0;
}