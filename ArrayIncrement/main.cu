#include <stdio.h>

__global__ void increment_gpu(int *a, int N) {
    int i = threadIdx.x;
    if (i < N) 
        a[i] += 1;
}

int main() {
    const int N = 5;
    int h_a[N] = {1, 2, 3, 4, 5};

    int *d_a;
    cudaMalloc(&d_a, N * sizeof(int));
    cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);

    dim3 grid_size(1); 
    dim3 block_size(N);

    increment_gpu<<<grid_size, block_size>>>(d_a, N);

    cudaMemcpy(h_a, d_a, N * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        printf("%d ", h_a[i]);
    }
    printf("\n");

    cudaDeviceSynchronize();

    free(h_a);
    cudaFree(d_a);
}
