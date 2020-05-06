#include "util.h"

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_OMP
void extractOffsets(unsigned *locOfAgents, unsigned *locationListOffsets, unsigned length, unsigned nLocs) {
    locationListOffsets[0] = 0;
    #pragma omp parallel for
    for (unsigned i = 1; i < length; i++) {
        if (locOfAgents[i-1] != locOfAgents[i]) {
            for (unsigned j = locOfAgents[i-1]+1; j <= locOfAgents[i]; j++) {
                locationListOffsets[j] = i;
            }
        }
    }
    locationListOffsets[nLocs] = length;
}
#elif THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
//Need a kernel here
#endif
void Util::updatePerLocationAgentLists(const thrust::device_vector<unsigned> &locationOfAgents,
                                                  thrust::device_vector<unsigned> &locationAgentList,
                                                  thrust::device_vector<unsigned> &locationListOffsets) {
    //Make a copy of locationOfAgents
    thrust::device_vector<unsigned> locOfAgents(locationOfAgents);
    thrust::sequence(locationAgentList.begin(),locationAgentList.begin());
    //Now sort by location, so locationAgentList contains agent IDs sorted by location
    thrust::stable_sort_by_key(locOfAgents.begin(), locOfAgents.end(), locationAgentList.begin());
    //Now extract offsets into locationAgentList where locations begin
    unsigned *locOfAgentsPtr = thrust::raw_pointer_cast(locOfAgents.data());
    unsigned *locationListOffsetsPtr = thrust::raw_pointer_cast(locationListOffsets.data());
    extractOffsets(locOfAgentsPtr, locationListOffsetsPtr, locOfAgents.size(), locationListOffsets.size()-1);

};