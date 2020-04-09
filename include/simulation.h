#pragma once
#include "agentsList.h"
#include "location.h"

template<typename PPState, typename PositionType, typename TypeOfLocation>
class Simulation {
    using LocationType = Location<PositionType, TypeOfLocation>;

    std::vector<LocationType> locations;
    AgentList<PPState, LocationType>* agents = AgentList<PPState, LocationType>::getInstance();

    // We can make it to a singleton later, but who knows
public:
    AgentList<PPState, LocationType>* agentList() { return agents; }
    // TODO add addLocation function instead
    std::vector<LocationType>& locationList() { return locations; }
};