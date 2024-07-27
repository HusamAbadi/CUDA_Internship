#include <stdio.h>
#include <cuda.h>

int main() {
    int dev_count = 0;  // Initialize dev_count to avoid warnings
    cudaError_t err = cudaGetDeviceCount(&dev_count);
    if (err != cudaSuccess) {
        printf("cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    cudaDeviceProp dev_prop;
    for (int i = 0; i < dev_count; i++) {
        err = cudaGetDeviceProperties(&dev_prop, i);
        if (err != cudaSuccess) {
            printf("cudaGetDeviceProperties failed for device %d: %s\n", i, cudaGetErrorString(err));
            continue;
        }

        printf("Device %d: %s\n", i, dev_prop.name);
        printf("  Total Global Memory: %zu bytes\n", dev_prop.totalGlobalMem);
        printf("  Shared Memory per Block: %zu bytes\n", dev_prop.sharedMemPerBlock);
        printf("  Registers per Block: %d\n", dev_prop.regsPerBlock);
        printf("  Warp Size: %d\n", dev_prop.warpSize);
        printf("  Max Threads per Block: %d\n", dev_prop.maxThreadsPerBlock);
        printf("  Max Threads Dimension: [%d, %d, %d]\n",
               dev_prop.maxThreadsDim[0], dev_prop.maxThreadsDim[1], dev_prop.maxThreadsDim[2]);
        printf("  Max Grid Size: [%d, %d, %d]\n",
               dev_prop.maxGridSize[0], dev_prop.maxGridSize[1], dev_prop.maxGridSize[2]);
        printf("  Clock Rate: %d kHz\n", dev_prop.clockRate);
        printf("  Total Constant Memory: %zu bytes\n", dev_prop.totalConstMem);
        printf("  Compute Capability: %d.%d\n", dev_prop.major, dev_prop.minor);
        // Add other properties as needed
    }

    return 0;
}
