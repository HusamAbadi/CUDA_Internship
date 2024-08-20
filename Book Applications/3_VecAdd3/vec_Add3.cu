// ! Using cudaMallocManaged()
//? Explanation: Unified Memory (cudaMallocManaged): The cudaMallocManaged() function is used instead of cudaMalloc() to allocate memory that can be accessed by both the CPU and GPU without the need for explicit memory copying. Data Transfer: Since cudaMallocManaged() creates a unified memory space, there's no need for cudaMemcpy() to transfer data between the host and the device. The host data is directly copied into the unified memory using std::copy. Synchronization: cudaDeviceSynchronize() is called after the kernel execution to ensure that the GPU has finished its work before the host accesses the data. Free Memory: cudaFree() is still used to free the memory allocated by cudaMallocManaged(). 
//* This approach simplifies memory management and can improve development speed, especially when debugging, because it eliminates the need for explicit memory copies. However, performance might be lower compared to manual memory management, depending on the specific use case.

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
    
    // Initializing the vectors with random numbers in each array between 0-100
    std::generate(begin(a), end(a), []() { return rand() % 100; });
    std::generate(begin(b), end(b), []() { return rand() % 100; });
    
    // Allocate unified memory accessible by both CPU and GPU (u_ : unified)
    int *u_a, *u_b, *u_c;
    cudaMallocManaged(&u_a, bytes);
    cudaMallocManaged(&u_b, bytes);
    cudaMallocManaged(&u_c, bytes);
    
    // Copy host data to unified memory (no need for cudaMemcpy)
    std::copy(a.begin(), a.end(), u_a);
    std::copy(b.begin(), b.end(), u_b);
    
    
    // Thread per CTA (256 Threads)
    int threads = 1 << 8;
    // Blocks per Grid (1 Block)
    int blocks = (int)ceil(N / threads);
    
    // Setup our kernel launch parameters
    dim3 block(threads, 1, 1);
    dim3 grid(blocks, 1, 1);
    
    // Launch our kernel
    vecAddKernel<<<grid, block>>>(u_a, u_b, u_c, N);
    
    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();
    
    // Copy results back to host vectors (optional, can use u_c directly)
    std::vector<int> c(u_c, u_c + N);

    // Display results on the terminal
    for(int i = 0; i < N; i++){
        std::cout << "a[" << i << "] + b[" << i << "] = " << a[i] << " + " << b[i] << " = " << u_c[i] << std::endl;
    }

    // Free memory
    cudaFree(u_a);
    cudaFree(u_b);
    cudaFree(u_c);

    std::cout << "Success!" << std::endl;

    return 0;
}