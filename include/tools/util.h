#pragma once
#include "datatypes.h"

class Util {
    public:
    static void updatePerLocationAgentLists(const thrust::device_vector<unsigned> &locationOfAgents,
                                                  thrust::device_vector<unsigned> &locationIdsOfAgents,
                                                  thrust::device_vector<unsigned> &locationAgentList,
                                                  thrust::device_vector<unsigned> &locationListOffsets);
};

template<typename UnaryFunction,
         typename PPState_t>
void reduce_by_location(thrust::device_vector<unsigned> &locationListOffsets, 
                        thrust::device_vector<unsigned> &fullInfectedCounts,
                        thrust::device_vector<PPState_t> &PPValues,
                        UnaryFunction lam) {
unsigned numLocations = locationListOffsets.size()-1;
unsigned *locationListOffsetsPtr = thrust::raw_pointer_cast(locationListOffsets.data());
unsigned *fullInfectedCountsPtr = thrust::raw_pointer_cast(fullInfectedCounts.data());
PPState_t *PPValuesPtr = thrust::raw_pointer_cast(PPValues.data());
    
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_OMP
#pragma omp parallel for
for (unsigned l = 0; l < numLocations; l++) {
    for (unsigned agent = locationListOffsetsPtr[l]; agent < locationListOffsetsPtr[l+1]; agent++) {
        fullInfectedCountsPtr[l] += lam(PPValuesPtr[agent]);
    }
}
#elif THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
//Need a kernel here
#endif
}