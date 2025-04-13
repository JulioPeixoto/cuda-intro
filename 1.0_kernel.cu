
// kernel definition
__global__ void vector_sums(int *a, int *b, int *c, int n){
    int = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// blockIdx is the block index
// blockDim is the number of threads per block
// threadIdx is the thread index inside the block

// kernel call
vector_sums<<<1, 1024>>>(a, b, c, n);

// copy result to host
cudaMemcpy(c_host, c_device, n * sizeof(int), cudaMemcpyDeviceToHost);
