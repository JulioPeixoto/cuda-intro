// thread block clusters example
// this example shows how to use thread block clusters to reduce the sum of an array

#include <stdio.h>
#include <cooperative_groups.h>

#define CLUSTER_SIZE 4
#define BLOCK_SIZE 32

namespace cg = cooperative_groups;


__global__ void __cluster_dims__(CLUSTER_SIZE, 1, 1) 
cluster_reduce_sum(int n, float *arr, float *sum) {
    __shared__ float shared_mem[BLOCK_SIZE];
    __shared__ float cluster_sum;

    cg::cluster_group cluster = cg::this_cluster();
    unsigned int cluster_block_rank = cluster.block_rank();
    unsigned int cluster_size = cluster.dim_blocks().x;

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    shared_mem[threadIdx.x] = 0.0f;
    if (idx < n) {
        shared_mem[threadIdx.x] = arr[idx];
    }

    __syncthreads();

    for (int offset = BLOCK_SIZE / 2; offset; offset /= 2) {
        if (threadIdx.x < offset) { 
            shared_mem[threadIdx.x] += shared_mem[threadIdx.x + offset];
        }
        __syncthreads();
    }

    cluster.sync();

    if (threadIdx.x == 0 && cluster_block_rank == 0) {
        atomicAdd(sum, cluster_sum);
    }
    
    cluster.sync();

} 
