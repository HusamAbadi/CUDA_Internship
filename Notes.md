<!--? Definitions: -->
    <!--* Thread: --> 
    A thread is the basic unit of execution in CUDA programming, representing an independent path of execution that runs on a single core of the GPU. In the context of CUDA, threads execute the same program, known as a kernel, but each thread processes its own subset of the data.

    <!--* Block:-->
    A block is a group of threads that execute the kernel code concurrently. Threads within a block can communicate with each other using shared memory and can synchronize their execution. A block is executed by a single Streaming Multiprocessor (SM) on the GPU.

    <!--* Grid:-->
    A grid is a collection of blocks that execute a CUDA kernel. The grid can consist of one or more blocks, allowing the kernel to scale across many threads and blocks, which in turn are distributed across the GPU's SMs.

    <!--* Streaming Multiprocessors (SMs):-->
    SMs are the main processing units within a CUDA-capable GPU. Each SM contains multiple CUDA cores (which execute threads), shared memory, and other resources. Each block of threads is assigned to a single SM for execution.

    <!--* Warp:-->
    A warp is a group of 32 threads within a block that are executed simultaneously by an SM. Warps are the smallest unit of execution on a GPU, and all threads in a warp execute the same instruction at the same time, but on different data.

    <!--* Stub Function:-->
    Stub Function: In CUDA, a stub function is a placeholder function that launches the kernel on the GPU. It's essentially a host function that sets up the grid and block dimensions and invokes the kernel on the device.

    <!--* Kernel:-->
    Kernel: A kernel is a function written in CUDA C/C++ that runs on the GPU. The kernel is executed by many threads in parallel, and each thread runs the kernel code with its own unique thread index.

<!--? Terminologies: -->
    <!--* Transparent Scalability:--> 
    Transparent scalability refers to the ability of a CUDA program to scale its execution across different GPUs with varying numbers of cores, SMs, and other resources without requiring changes to the code. This means the same CUDA program can run efficiently on a wide range of NVIDIA GPUs.

    <!--* Autotuning:-->
    Autotuning is the process of automatically optimizing the performance of a CUDA kernel by exploring different configurations of parameters, such as block size, grid size, or memory usage, to find the most efficient settings for a particular hardware setup.

    <!--* Latency Tolerance -->
    The mechanism of filling the latency time of operations with work from other threads.

<!--? Consepts: -->
    <!--* Memory Space:-->
    In CUDA, memory space refers to the different types of memory available on the GPU, including global memory, shared memory, constant memory, and registers. Each memory space has its own characteristics, such as size, speed, and scope, and is used for different purposes in CUDA programming.

    <!--* Pixel Shader:-->
    A pixel shader is a type of GPU program used in graphics processing to compute effects on a per-pixel basis. It is used to control the color and other attributes of individual pixels in rendered images, allowing for complex visual effects like lighting, shading, and texturing.

<!--? Important Information -->
A block can have up to 1024 threads.
An SP is designed to execute one thread from a warp at a time. Since a warp consists of 32 threads, an SP can process all threads in a warp over multiple clock cycles. 

The GPU is responisble for allocating the blocks to the SMs