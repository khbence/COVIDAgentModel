#pragma once
#include "agentsList.h"
#include "location.h"

template<typename PPState, typename PositionType, typename LocationType>
class Simulation {
    std::vector<Location<PositionType, LocationType>> locations;
    AgentList<PPState, decltype(locations)::value_type> agents;

    //We can make it to a singleton later
};