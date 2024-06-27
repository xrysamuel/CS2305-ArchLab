#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>

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

__global__ void transposeNaive(float *inputMat, float *outputMat, int N)
{
    int in_x = blockIdx.x * blockDim.x + threadIdx.x;
    int in_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (in_x < N && in_y < N)
    {
        outputMat[in_y * N + in_x] = inputMat[in_x * N + in_y];
    }
}

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


#define N_IMP 11

void transposeGPUWrapper(float *inputMat, float *outputMat, int N, int imp)
{
    if (imp == 0)
    {
        dim3 blockSize(16, 16);
        dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);
        transposeNaive<<<gridSize, blockSize>>>(inputMat, outputMat, N);
    }
    else if (imp == 1)
    {
        dim3 blockSize(8, 8);
        dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);
        transposeShared<8><<<gridSize, blockSize>>>(inputMat, outputMat, N);
    }
    else if (imp == 2)
    {
        dim3 blockSize(16, 16);
        dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);
        transposeShared<16><<<gridSize, blockSize>>>(inputMat, outputMat, N);
    }
    else if (imp == 3)
    {
        dim3 blockSize(32, 32);
        dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);
        transposeShared<32><<<gridSize, blockSize>>>(inputMat, outputMat, N);
    }
    else if (imp == 4)
    {
        dim3 blockSize(8, 8);
        dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);
        transposeSharedwBC<8><<<gridSize, blockSize>>>(inputMat, outputMat, N);
    }
    else if (imp == 5)
    {
        dim3 blockSize(16, 16);
        dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);
        transposeSharedwBC<16><<<gridSize, blockSize>>>(inputMat, outputMat, N);
    }
    else if (imp == 6)
    {
        dim3 blockSize(32, 32);
        dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);
        transposeSharedwBC<32><<<gridSize, blockSize>>>(inputMat, outputMat, N);
    }
    else if (imp == 7)
    {
        dim3 blockSize(32, 16);
        dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);
        transposeUnrolled<32, 16><<<gridSize, blockSize>>>(inputMat, outputMat, N);
    }
    else if (imp == 8)
    {
        dim3 blockSize(32, 8);
        dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);
        transposeUnrolled<32, 8><<<gridSize, blockSize>>>(inputMat, outputMat, N);
    }
    else if (imp == 9)
    {
        dim3 blockSize(64, 16);
        dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);
        transposeUnrolled<64, 16><<<gridSize, blockSize>>>(inputMat, outputMat, N);
    }
    else if (imp == 10)
    {
        cudaMemcpy(inputMat, outputMat, N * N * sizeof(float), cudaMemcpyDeviceToDevice);
    }
}

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
    transposeGPUWrapper(inputMatGPU, outputMatGPU, N, imp); // warnup

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

float measureCPUTime(float *inputMat, float *outputMat, int N)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    transposeCPU(inputMat, outputMat, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    return milliseconds;
}

bool verifyTranspose(float *matrixA, float *matrixB, int N)
{
    for (int i = 0; i < N * N; i++)
    {
        if (matrixA[i] != matrixB[i])
        {
            return false;
        }
    }
    return true;
}

int main()
{
    int N = 4000;

    float *inputMat = new float[N * N];
    float *outputMatTrue = new float[N * N];
    float *outputMat = new float[N * N];

    for (int i = 0; i < N * N; i++)
    {
        inputMat[i] = i;
    }

    float time;

    time = measureCPUTime(inputMat, outputMatTrue, N);
    std::cout << "CPU time: " << time << "ms" << std::endl;

    for (int imp = 0; imp < N_IMP; imp++)
    {
        time = measureGPUTime(inputMat, outputMat, N, imp);
        if (!verifyTranspose(outputMatTrue, outputMat, N))
            std::cout << "Wrong!" << std::endl;
        std::cout << imp << " GPU time: " << time << "ms" << std::endl;
    }

    delete[] inputMat;
    delete[] outputMat;
    delete[] outputMatTrue;

    return 0;
}