#include <stdio.h>
#include <cuda.h>

// Kernel function to process the image
__global__ 
void pictureKernel(float* d_Pin, float* d_Pout, int n, int m) {
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    if (Row < m && Col < n) {
        d_Pout[Row * n + Col] = 2 * d_Pin[Row * n + Col];
    }
}

// Host function to set up and launch the kernel
void picture(float* h_Pin, float* h_Pout, int n, int m) {
    int size = n * m * sizeof(float);
    float *d_Pin, *d_Pout;

    // Allocate device memory
    cudaMalloc((void **)&d_Pin, size);
    cudaMalloc((void **)&d_Pout, size);

    // Copy data from host to device
    cudaMemcpy(d_Pin, h_Pin, size, cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 dimBlock(16, 16, 1);
    dim3 dimGrid((n + dimBlock.x - 1) / dimBlock.x, (m + dimBlock.y - 1) / dimBlock.y, 1);

    // Launch the kernel
    pictureKernel<<<dimGrid, dimBlock>>>(d_Pin, d_Pout, n, m);

    // Copy the result back to the host
    cudaMemcpy(h_Pout, d_Pout, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_Pin);
    cudaFree(d_Pout);
}

int main() {
    int n = 76;
    int m = 62;

    // Allocate host memory
    float *h_Pin = (float *)malloc(n * m * sizeof(float));
    float *h_Pout = (float *)malloc(n * m * sizeof(float));

    // Initialize the input array with some values (for example, set all to 1.0)
    for (int i = 0; i < n * m; i++) {
        h_Pin[i] = 1.0f;
    }

    // Call the picture function
    picture(h_Pin, h_Pout, n, m);

    // Print some of the output values to check
    for (int i = 0; i < n*m ; i++) {
        printf("h_Pin[%d] = %f", i, h_Pin[i]);
        printf (" --- ");
        printf("h_Pout[%d] = %f\n", i, h_Pout[i]);
    }

    // Free host memory
    free(h_Pin);
    free(h_Pout);

    return 0;
}
