#pragma once
#include <vector>
#include "globalStates.h"

// concept
template<typename PositionType, typename TypeOfLocation, typename Statistics>
class Location : public Statistics {
    PositionType position;
    TypeOfLocation locType;
    std::vector<unsigned> agents;

public:
    Location(PositionType p, TypeOfLocation t) : position(p), locType(t) {}
    void addAgent(unsigned a) { agents.push_back(a); }
};