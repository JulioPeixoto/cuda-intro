#include <cuda_runtime.h>
#include <stdio.h>

__global__ void increment_kernel(int *data, int value, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] += value;
    }
}

int main() {
    const int dataSize = 1 << 20;
    const int dataBytes = dataSize * sizeof(int);

    int *hostData;
    int *deviceData;

    cudaStream_t stream;

    //Memory allocation on host
    cudaMallHost(&hostData, dataBytes);

    for (int i = 0; i < dataSize; ++i) {
        hostData[i] = i;
    }

    // Memory allocation on device
    cudaMalloc(&deviceData, dataBytes);

    // Stream creation
    cudaStreamCreate(&stream);

    // Asynchronous copy from host to device
    cudaMemcpyAsync(deviceData, hostData, dataBytes, cudaMemcpyAsyncHostToDevice, stream);

    int threadsPerBlock = 256;
    int blocksPerGrid = (dataSize + threadsPerBlock - 1) / threadsPerBlock;
    increment_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(deviceData, 1, dataSize);

    // Asynchronous copy from device to host
    cudaMemcpyAsync(hostData, deviceData, dataBytes, cudaMemcpyAsyncDeviceToHost, stream);

    // Synchronize the stream
    cudaStreamSynchronize(stream);

    // Results verification
    bool success = true;
    for (int i = 0, i < dataSize; ++i) {
        if (hostData[i] != i + incrementValue) {
            success = false;
            printf("Erro na posição %d: %d != %d\n", i, hostData[i], i + incrementValue);
            break;
        }
    }

    if (success) {
        printf("Operação assíncrona concluída com sucesso!\n");
    } 

    // Memory release and stream destruction
    cudaStreamDestroy(stream);
    cudaFreeHost(hostData);
    cudaFree(deviceData);

    return 0;
}
