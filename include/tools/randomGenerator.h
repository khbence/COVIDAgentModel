#pragma once
#include <random>
#include <vector>
#include <omp.h>
#include "datatypes.h"
#include "timing.h"

#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
#include <cuda.h>
#include <curand_kernel.h>
__device__ curandState *dstates;
__global__ void setup_kernel(unsigned total)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread gets same seed, a different sequence 
       number, no offset */
    if (id < total)
        curand_init(1234, id, 0, &dstates[id]);
}
#endif

class RandomGenerator {
    #if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_OMP
    static std::vector<std::mt19937_64> generators;
    #endif

public:
    static void init(unsigned threads) {
    #if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
    curandState *devStates;
    cudaMalloc((void **)&devStates, threads * 
                  sizeof(curandState));
    cudaMemcpyToSymbol(dstates, devStates, threads * 
                  sizeof(curandState));
    setup_kernel<<<(threads-1)/128+1,128>>>(threads);
    cudaDeviceSynchronize();
    #elif THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_OMP
        generators.reserve(threads);
        std::random_device rd;
        for (unsigned i = 0; i < threads; ++i) { generators.emplace_back(rd()); }
    #endif
    }

    [[nodiscard]] static thrust::host_vector<float> fillUnitf(unsigned size) {
        #if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_OMP
        Timing::startTimer("RandomGenerator::fillUnitf");
        thrust::host_vector<float> tmp(size);
        std::uniform_real_distribution<double> dis(0, 1);
        //#pragma omp parallel for private(dis)
        for (int i = 0; i < size; i++) {
            tmp[i] = dis(generators[0]);
        }
        Timing::stopTimer("RandomGenerator::fillUnitf");
        return tmp;
        #endif
    }

    [[nodiscard]] static HD double randomUnit() {
        #if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
        return curand_uniform_double(&dstates[threadIdx.x+blockIdx.x*blockDim.x]);
        #elif THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_OMP
        std::uniform_real_distribution<double> dis(0, 1);
        return dis(generators[omp_get_thread_num()]);
        #endif
    }

    [[nodiscard]] static HD double randomReal(double max) {
        #if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
        return max*curand_uniform_double(&dstates[threadIdx.x+blockIdx.x*blockDim.x]);
        #elif THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_OMP
        std::uniform_real_distribution<double> dis(0, max);
        return dis(generators[omp_get_thread_num()]);
        #endif
    }

    [[nodiscard]] static HD unsigned randomUnsigned(unsigned max) {
        #if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
        return curand(&dstates[threadIdx.x+blockIdx.x*blockDim.x])%(max+1);
        #elif THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_OMP
        std::uniform_int_distribution<unsigned> dis(0, max);
        return dis(generators[omp_get_thread_num()]);
        #endif
    }

    [[nodiscard]] static HD int geometric(double p) {
        #if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
        //TODO: need geometric here!!!
        return curand_uniform_double(&dstates[threadIdx.x+blockIdx.x*blockDim.x]);
        #elif THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_OMP
        std::geometric_distribution<> dis(p);
        return dis(generators[omp_get_thread_num()]);
        #endif
    }
};