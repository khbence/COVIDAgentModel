#pragma once
#include <vector>

template<typename PositionType, typename TypeOfLocation>
class Location {
    PositionType position;
    TypeOfLocation locType;
    std::vector<unsigned> agents;

public:
};