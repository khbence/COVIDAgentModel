#pragma once
#include "datatypes.h"
#include "timing.h"

class Util {
public:
    static void updatePerLocationAgentLists(const thrust::device_vector<unsigned>& locationOfAgents,
        thrust::device_vector<unsigned>& locationIdsOfAgents,
        thrust::device_vector<unsigned>& locationAgentList,
        thrust::device_vector<unsigned>& locationListOffsets);
};

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
// TODO: swap this for loop over agents and atomicAdd to locaiton
template<typename UnaryFunction, typename Count_t, typename PPState_t>
__global__ void reduce_by_location_kernel(unsigned* locationListOffsetsPtr,
    Count_t* fullInfectedCountsPtr,
    PPState_t* PPValuesPtr,
    unsigned numLocations,
    UnaryFunction lam) {
    unsigned l = threadIdx.x + blockIdx.x * blockDim.x;
    if (l < numLocations) {
        for (unsigned agent = locationListOffsetsPtr[l]; agent < locationListOffsetsPtr[l + 1];
             agent++) {
            fullInfectedCountsPtr[l] += lam(PPValuesPtr[agent]);
        }
    }
}
#endif
template<typename UnaryFunction, typename Count_t, typename PPState_t>
void reduce_by_location(thrust::device_vector<unsigned>& locationListOffsets,
    thrust::device_vector<Count_t>& fullInfectedCounts,
    thrust::device_vector<PPState_t>& PPValues,
    UnaryFunction lam) {
    unsigned numLocations = locationListOffsets.size() - 1;
    unsigned* locationListOffsetsPtr = thrust::raw_pointer_cast(locationListOffsets.data());
    Count_t* fullInfectedCountsPtr = thrust::raw_pointer_cast(fullInfectedCounts.data());
    PPState_t* PPValuesPtr = thrust::raw_pointer_cast(PPValues.data());

    PROFILE_FUNCTION();

    if (numLocations == 1) {
        fullInfectedCounts[0] =
            thrust::reduce(thrust::make_transform_iterator(PPValues.begin(), lam),
                thrust::make_transform_iterator(PPValues.end(), lam),
                (Count_t)0.0f);
    } else {

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_OMP
#pragma omp parallel for
        for (unsigned l = 0; l < numLocations; l++) {
            for (unsigned agent = locationListOffsetsPtr[l]; agent < locationListOffsetsPtr[l + 1];
                 agent++) {
                fullInfectedCountsPtr[l] += lam(PPValuesPtr[agent]);
            }
        }
#elif THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
        reduce_by_location_kernel<<<(numLocations - 1) / 256 + 1, 256>>>(
            locationListOffsetsPtr, fullInfectedCountsPtr, PPValuesPtr, numLocations, lam);
        cudaDeviceSynchronize();
#endif
    }
}