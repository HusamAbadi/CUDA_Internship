#include <stdio.h>
#include <cstdlib>
#include <cassert>
#include <iostream>

using namespace std;

__global__ void matrixMul(int* a, int* b, int* c, int N){
    // cALCULATE THE GLOBAL ROW AND COLUMN FOR EACH THREAD
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // CHECK IF OUT OF BOUNDS
    if (row < N && col < N) {
        int sum = 0;
        for (int i = 0; i < N; i++) {
            sum += a[row * N + i] * b[i * N + col];
        }
        // Write back the result to global memory
        c[row * N + col] = sum;
    }
}

// Initializes a square matrix with random numbers between 0-100
void init_matrix(int *m, int N){
    for (int i = 0; i < N * N; i++) {
        m[i] = rand() % 100;
    }
}

// Verif the result on the CPU
void verify_result(int *a, int *b, int *c, int N){
    int sum;
    // for every row
    for (int i = 0; i < N; i++) {
        // for every column
        for (int j = 0; j < N; j++) {
            // for every element in the row-column pair
            sum = 0;
            for (int k = 0; k < N; k++) {
                sum += a[i * N + k] * b[k * N + j];
             // sum += a[row * N + i] * b[i * N + col];
            }
            // check if the calculated sum is correct
            assert(sum == c[i * N + j]);
        }
    }
}

int main(){
    // Get our device ID for other CUDA calls
    int id = cudaGetDevice(&id);

    // Set our square matrix dimensions (2^10 x 2^10 default = 2^20) = (1,024 x 1,024) = (1,048,576)
    int N = 1 << 10; // 2^10 = 1024
    size_t bytes = N * N * sizeof(int);

    // Allocate memory for our matrices
    int *a, *b, *c;
    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);
    cudaMallocManaged(&c, bytes);
    // cudaMallocManaged: This line of code allocates bytes amount of memory on the GPU and makes it accessible to both the GPU and the host (CPU) using a single pointer a. This is done using CUDA's Unified Memory feature, which allows for easier memory management between the host and device.

    // Initialize our matrices
    init_matrix(a, N);
    init_matrix(b, N);

    // Set our block/(CTA: Cooperative thread array) and grid dimensions
    int threads = 32;
    int blocks = (int)ceil(N / (float)threads);
    //* OR
    blocks = (N + threads - 1) / threads; //* Preferred

    // Setup our kernel launch parameters
    dim3 block(threads, threads, 1); // (32, 32, 1) = 1,024 Threads
    dim3 grid(blocks, blocks, 1); // (32, 32, 1) = 1,024 Blocks

    // Launch our kernel
    //* Uncomment these for pre-fetching 'a' and 'b' vectors to device memory (To enhance performance)
    cudaMemPrefetchAsync(a, bytes, id);
    cudaMemPrefetchAsync(b, bytes, id);

    matrixMul<<<grid, block>>>(a, b, c, N);
    
    // if we don't use memCopy() we need to call cudaDeviceSynchronize().
    cudaDeviceSynchronize();

    cudaMemPrefetchAsync(c, bytes, cudaCpuDeviceId);


    // Verify our results
    // verify_result(a, b, c, N);
    cout << "Success!" << endl;
    return 0;
}