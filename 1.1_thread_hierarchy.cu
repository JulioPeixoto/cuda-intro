__global__ void thread_hierarchy(float *a, float *b, float *c, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        c[index] = a[index] + b[index];
    }
}


// each thread calculates the sum of a corresponding element in the A and B matrices. 
// two-dimensional indexing facilitates direct mapping between threads and matrix elements.