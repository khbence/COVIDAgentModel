#include "randomGenerator.h"

std::vector<std::mt19937_64> RandomGenerator::generators;

#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
#include <cuda.h>
#include <curand_kernel.h>
__device__ curandState *dstates;
__global__ void setup_kernel(unsigned total, curandState *dstates2)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread gets same seed, a different sequence 
       number, no offset */
    if (id < total)
        curand_init(1234, id, 0, &dstates2[id]);
}
#endif

void RandomGenerator::init(unsigned agents) {
        #if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
        curandState *devStates;
        cudaMalloc((void **)&devStates, agents * 
                    sizeof(curandState));
        cudaMemcpyToSymbol(dstates, &devStates, 
                    sizeof(curandState*));
        setup_kernel<<<(agents-1)/128+1,128>>>(agents,devStates);
        cudaDeviceSynchronize();
        #endif
        unsigned threads = omp_get_max_threads();
        generators.reserve(threads);
        std::random_device rd;
        for (unsigned i = 0; i < threads; ++i) { generators.emplace_back(rd()); }
    }