Ex 4.1:
    SM => 1536 Threads, 4 Blocks.
    Most number of threads cnfiguration: B) 256 threads per block.

Ex 4.2:
    N = 2000, block = 512 threads.
    How many threads in the grid: C) ceil(2000 / 512) = 4 => 4 * 512 = 2048 threads in the grid.

Ex 4.3:

Ex 4.4:
    400 X 900 image, squared thread blocks, use the maximum numbeer of threads per block (1024) (Capability 3.0):
    To have a squared thread blocks with the maximum number of threads per block we can use 32 X 32 Blocks.
    For the X dimension we can use 400 / 32 = 12.5 Blocks, and for the Y dimension we can use 900 / 32 = 28.15 Blocks.
    Which means we will end up having 13 X 29 Blocks = 377 Blocks. 

Ex 4.5:
    Number of Idle threads:
    (13 * 32) - 400 = 416 - 400 = 16 Idle Threads in the X dimension.
    (29 * 32) - 900 = 928 - 900 = 28 Idle Threads in the Y dimension.
    Total of 16 + 28 = 44 Idle Threads.

Ex 4.6:
    2.0, 2.3, 3.0, 2.8, 2.4, 1.9, 2.6, 2.9 => Percentage of the threads' summed-up execution times is spent waiting for the barrier.
    Answer: (sum of the execution time of each / sum-up of (3.0 - each time):)
    1.0 + 0.7 + 0.0 + 0.2 + 0.6 + 1.1 + 0.4 + 0.1 = 4.1 microseconds
    19.9 / 4.1 = 

