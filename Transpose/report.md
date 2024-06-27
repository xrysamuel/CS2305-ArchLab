# Matrix Transpose

## 1 Implementation of Matrix Transpose

In this experiment, we aim to implement the matrix transpose operation. The goal is to transpose an $N \times N$ single-precision floating-point matrix, with the input and output stored in two separate arrays in memory. We will implement different versions of matrix transpose and compare their performance.

### 1.1 CPU Implementation of Matrix Transpose

On the CPU, we employ the simplest method for matrix transpose, which involves a nested loop and enabling the `-O2` optimization. Its time complexity is $O(n^2)$.

```c
void transposeCPU(float *inputMat, float *outputMat, int N)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            outputMat[j * N + i] = inputMat[i * N + j];
        }
    }
}
```

### 1.2 GPU Implementation of Matrix Transpose

```c
__global__ void transposeNaive(float *inputMat, float *outputMat, int N)
{
    int in_x = blockIdx.x * blockDim.x + threadIdx.x;
    int in_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (in_x < N && in_y < N)
    {
        outputMat[in_y * N + in_x] = inputMat[in_x * N + in_y];
    }
}
```

In this naive implementation of matrix transpose, each thread is responsible for moving an element from `inputMat` to the corresponding position in `outputMat`. Specifically, in the kernel function, we first calculate the element indices `in_x` and `in_y` that the current thread is responsible for in the input matrix. Then, we copy this element to the output matrix. If all threads execute in parallel, the time complexity will be $O(1)$.

In theory, this approach can achieve optimal performance. However, when considering the characteristics of Global Memory, we find that threads within the same thread block read from consecutive positions in Global Memory, while writing to positions at intervals. We want the write positions in Global Memory to be relatively contiguous to enable more coalesced memory accesses. To achieve this, we can use Shared Memory for temporary data transfer.

### 1.3 Optimization Using Shared Memory

```c
template <int BLOCK_SIZE>
__global__ void transposeShared(float *inputMat, float *outputMat, int N)
{
    __shared__ float buffer[BLOCK_SIZE][BLOCK_SIZE];
    int out_block_x = blockIdx.x * BLOCK_SIZE;
    int out_block_y = blockIdx.y * BLOCK_SIZE;
    int in_x = out_block_y + threadIdx.y;
    int in_y = out_block_x + threadIdx.x;
    int out_x = out_block_x + threadIdx.y;
    int out_y = out_block_y + threadIdx.x;

    if (in_x < N && in_y < N)
    {
        buffer[threadIdx.x][threadIdx.y] = inputMat[in_x * N + in_y];
    }
    __syncthreads();
    if (out_x < N && out_y < N)
    {
        outputMat[out_x * N + out_y] = buffer[threadIdx.y][threadIdx.x];
    }
}
```

In this version of matrix transpose implementation, we first declare an array in Shared Memory. This array is shared among threads within a thread block and has the same lifespan as the block. Then, similar to the naive matrix transpose implementation, we calculate the elements that each thread block and thread need to process.

Blocked matrix transpose is based on the mathematical fact that if a matrix can be blocked as $A = (A_{ij})$, its transpose is $A^T = (A^T_{ji})$. Therefore, each thread block loads a blocked portion of the input matrix from Global Memory into Shared Memory and writes back this blocked portion in the order of the transpose to the corresponding position in the output matrix in Global Memory. It is important to note that `__syncthreads();` must be inserted after loading into Shared Memory to ensure that all threads in the block have finished loading before proceeding to the write-back operation.

This approach ensures that the block's access to Global Memory is localized.

### 1.4 Considering Bank Conflicts

For a Shared Memory region with $32 \times 32$ elements, all elements in a column are mapped to the same bank, resulting in bank conflicts: reading a column data causes 32-way bank conflicts. To avoid this, we can increase the number of columns in the Shared Memory region from 32 to 33, so that the elements in a column will not reside in the same bank.

Here, we adopt an alternative approach to avoid having the same column data in the same bank by staggering the elements in the same column.

```c
template <int BLOCK_SIZE>
__global__ void transposeSharedwBC(float *inputMat, float *outputMat, int N)
{
    __shared__ float buffer[BLOCK_SIZE][BLOCK_SIZE];
    int out_block_x = blockIdx.x * BLOCK_SIZE;
    int out_block_y = blockIdx.y * BLOCK_SIZE;
    int in_x = out_block_y + threadIdx.y;
    int in_y = out_block_x + threadIdx.x;
    int out_x = out_block_x + threadIdx.y;
    int out_y = out_block_y + threadIdx.x;

    if (in_x < N && in_y < N)
    {
        buffer[threadIdx.x][(threadIdx.x + threadIdx.y) % BLOCK_SIZE] = inputMat[in_x * N + in_y];
    }
    __syncthreads();
    if (out_x < N && out_y < N)
    {
        outputMat[out_x * N + out_y] = buffer[threadIdx.y][(threadIdx.x + threadIdx.y) % BLOCK_SIZE];
    }
}
```

![](media/transpose.png)

### 1.5 Performing More Computations per Thread

In the previous implementation, each thread only performed computations for one element. However, in reality, the kernel function can execute 2 independent instructions in parallel. We can make a thread perform more computations by unrolling. If we set TILE to 32 and SIDE to 8, only $32 \times 8$ threads are needed in a block, and each thread is responsible for computing 4 elements, while the previous implementation would require $32 \times 32$ threads.

```c
template <int TILE, int SIDE>
__global__ void transposeUnrolled(float *inputMat, float *outputMat, int N)
{
    __shared__ float buffer[TILE][TILE];
    int out_block_x = blockIdx.x * TILE;
    int out_block_y = blockIdx.y * TILE;
    int in_x = out_block_y + threadIdx.y;
    int in_y = out_block_x + threadIdx.x;
    int out_x = out_block_x + threadIdx.y;
    int out_y = out_block_y + threadIdx.x;
    #pragma unroll
    for (int offset = 0; offset < TILE; offset += SIDE)
    {
        if (in_y < N && in_x + offset < N)
            buffer[threadIdx.y + offset][(threadIdx.x + threadIdx.y + offset) % TILE] = inputMat[(in_x + offset) * N + in_y];
    }
    __syncthreads();
    #pragma unroll
    for (int offset = 0; offset < TILE; offset += SIDE)
    {
        if (out_y < N && out_x + offset < N)
           outputMat[(out_x + offset) * N + out_y] = buffer[threadIdx.x][(threadIdx.x + threadIdx.y + offset) % TILE];
    }
}
```

## 2 Experiment

### 2.1 Test Environment

The experiment was conducted on an NVIDIA A800-SXM4-80GB GPU with a peak FP32 performance of 19.5 TFLOPS. It has 40GB of high-speed HBM2e memory with a bandwidth of 1935 GB/s. The GPU has 128 streaming multiprocessors.

### 2.2 Test Code

Instead of using `<chrono>`, we use `cudaEventRecord` to measure GPU-side time because executing the kernel function may be asynchronous to the CPU. The `transposeGPUWrapper` in the code is a wrapper for the kernel function. Before measuring the runtime of the kernel function, we need to execute it once for warm-up to fully utilize the GPU's performance and accurately measure the real performance of different implementation versions.

```c
float measureGPUTime(float *inputMat, float *outputMat, int N, int imp)
{
    // malloc
    float *inputMatGPU;
    float *outputMatGPU;
    cudaMalloc((void **)&inputMatGPU, N * N * sizeof(float));
    cudaMalloc((void **)&outputMatGPU, N * N * sizeof(float));

    // copy inputMat from host to device
    cudaMemcpy(inputMatGPU, inputMat, N * N * sizeof(float), cudaMemcpyHostToDevice);

    // transpose
    transposeGPUWrapper(inputMatGPU, outputMatGPU, N, imp); // warm-up

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    transposeGPUWrapper(inputMatGPU, outputMatGPU, N, imp);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // copy outputMat from device to host
    cudaMemcpy(outputMat, outputMatGPU, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // free
    cudaFree(inputMatGPU);
    cudaFree(outputMatGPU);

    return milliseconds;
}
```

We conducted the test on a $4000 \times 4000$ ($N = 4000$) single-precision floating-point matrix. We executed the kernel function for each implementation version with different block sizes. We used the code above to measure the runtime and then checked the results to ensure they matched the results on the CPU.

We invoked the kernel function as follows:

```c
dim3 blockSize(16, 16);
dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);
transposeShared<16><<<gridSize, blockSize>>>(inputMat, outputMat, N);
```

### 2.3 Results

The experiment consisted of 12 cases:

- CPU implementation
- Naive: Naive GPU implementation
- Shared: GPU implementation using shared memory with block sizes of 8, 16, and 32
- SharedwBC: GPU implementation using shared memory considering bank conflicts with block sizes of 8, 16, and 32
- Unrolled: GPU implementation with loop unrolling using `(TILE, SIDE)` values of (32, 16), (32, 8), and (64, 16)
- memcpy: `cudaMemcpy(inputMat, outputMat, N * N * sizeof(float), cudaMemcpyDeviceToDevice);` used to determine the upper limit of matrix transpose optimization

The results are shown in the following graph:

![](media/res.png)

Firstly, let's analyze the impact of different block sizes on performance. In the Shared and SharedwBC implementation versions, the block size of 16 achieves the best performance. Additionally, for a block size of 32, the latency of SharedwBC is significantly reduced compared to Shared, indicating that bank conflicts have a huge impact on performance when the block size matches the number of banks. In the Unrolled implementation version, the lowest latency is achieved when TILE is 32 and SIDE is 16. 

Next, let's compare the best performance among different implementation versions:

|Implementation|Latency (ms)|Bandwidth (GB/s)|Speedup|
|:--:|:--:|:--:|:--:|
|CPU|50.4135|2.539|1|
|Naive|0.129024|992.06|391|
|Shared|0.122880|1041.67|410|
|SharedwBC|0.111616|1146.79|451|
|Unrolled|0.108544|1179.25|464|
|memcpy|0.086016|1488.10|586|


The results indicate that the optimization techniques we applied during the implementation process indeed improve performance. However, it's also important to note that choosing an appropriate block size may be more crucial than selecting the optimal implementation approach.

## 3 Conclusion

Matrix transpose is a fundamental operation in linear algebra and has various applications in numerical computations. Implementing matrix transpose efficiently can lead to performance improvements, especially when working with large matrices.

In this experiment, we compared different implementations of matrix transpose on the GPU. We found that using Shared Memory and avoiding bank conflicts can improve the performance of the matrix transpose operation. The specific performance improvement may vary depending on the GPU model and its specifications such as block size.

Optimizing matrix transpose is just one example of how parallel computing techniques can be applied to improve performance in numerical computations. By leveraging the power of GPUs and optimizing algorithms for parallel execution, we can achieve faster and more efficient computations in various domains.