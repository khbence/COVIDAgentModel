#pragma once
#include "datatypes.h"

class Util {
    public:
    static void updatePerLocationAgentLists(const thrust::device_vector<unsigned> &locationOfAgents,
                                                  thrust::device_vector<unsigned> &locationAgentList,
                                                  thrust::device_vector<unsigned> &locationListOffsets);
};