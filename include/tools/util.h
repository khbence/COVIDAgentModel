#pragma once
#include "datatypes.h"

class Util {
    static void updatePerLocationAgentLists(const thrust::device_vector<unsigned> &locationOfAgents,
                                                  thrust::device_vector<unsigned> &locationAgentList,
                                                  thrust::device_vector<unsigned> &locationListOffsets);
}