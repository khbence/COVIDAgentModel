#include "util.h"
#include "timing.h"

void extractOffsets(unsigned *locOfAgents, unsigned *locationListOffsets, unsigned length, unsigned nLocs) {
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_OMP
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
#elif THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
}
//Need a kernel here
#endif
void Util::updatePerLocationAgentLists(const thrust::device_vector<unsigned> &locationOfAgents,
                                                  thrust::device_vector<unsigned> &locationIdsOfAgents,
                                                  thrust::device_vector<unsigned> &locationAgentList,
                                                  thrust::device_vector<unsigned> &locationListOffsets) {
    PROFILE_FUNCTION();
    //Make a copy of locationOfAgents
    thrust::copy(locationOfAgents.begin(), locationOfAgents.end(), locationIdsOfAgents.begin());
    thrust::sequence(locationAgentList.begin(),locationAgentList.end());
    //Now sort by location, so locationAgentList contains agent IDs sorted by location
    BEGIN_PROFILING("sort")
    thrust::stable_sort_by_key(locationIdsOfAgents.begin(), locationIdsOfAgents.end(), locationAgentList.begin());
    END_PROFILING("sort")
    //Now extract offsets into locationAgentList where locations begin
    unsigned *locationIdsOfAgentsPtr = thrust::raw_pointer_cast(locationIdsOfAgents.data());
    unsigned *locationListOffsetsPtr = thrust::raw_pointer_cast(locationListOffsets.data());
    extractOffsets(locationIdsOfAgentsPtr, locationListOffsetsPtr, locationIdsOfAgents.size(), locationListOffsets.size()-1);
};