# Day 21 (Aug 14th):
#- Book Revision: pages 0-31 (I started revising the book from the beginning because I felt I was missing some points in the previous chapters.)

# Day 22 (Aug 15th):
#- Book Revision: pages 32-45

# Day 23 (Aug 16th):
#- Book Revision: pages 46-69

# Day 24 (Aug 17th):
#- Book Revision: pages 70-79
#- Video Watching: (From Scratch: Matrix Multiplication in CUDA - By CoffeeBeforeArch)

# Day 25 (Aug 18th):
#- Video Watching: Continued watching the last video and continued learning the concepts.
#- Practicing : I revised on the previous exercises and improved exercises of chapter 3 and 4.
    Started working on the matrix-matrix multiplication problem which was introduced in chapter 4.3 in the book.

# Day 26 (Aug 19th):
#- Video Watching: Completed watching the last video and followed it to the end.
#- Practicing : I finished working on the matrix-matrix multiplication problem and it worked successfully.
                    The program included a result verification function that checked if the result was correct by copmuting the multiplication on the CPU (It took the CPU several seconds to compute but miliseconds for the GPU).
                I created another Vector Addition program that uses a different approach in initializing the vectors and generating random numbers.
                    (In the output array the elements where added correctly up to the 98th index. The discrepancies start at index 99 and continue:)
# Day 28 (Aug 20th):
#- Book Revision: pages 80-86
#- Practicing : updated vecAdd1,2 and created vecAdd3 program that uses cudaMallocManaged() function.
                I also played around with the array inputs and kernels configuration parameters to test compiling and execution timings.
                I created a new vector addition program called vecAdd3 that uses the cudaMallocManaged() function.
                    I learnt how to use cudaMallocManaged() function in CUDA instead of using the normal cudaMalloc() and what I learnt is that The cudaMallocManaged() function is used instead of cudaMalloc() to allocate memory that can be accessed by both the CPU and GPU without the need for explicit memory copying.


# Day 29 (Aug 21st):
#- Book Revision: pages 86-89
#- Practicing : updated matrix_mul program kernel launch configuration parameters and Added prefetching functions to enhance performance.

# Day 30 (Aug 22nd):
#- Book Revision: pages 89-94
#- Practicing : Solved some exercises which I created a new folder for

