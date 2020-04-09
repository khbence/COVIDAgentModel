#pragma once
#include "LocationStats.h"

template<typename SimulationType>
class BasicInfection {
public:
    using StatisticType = LocationStats;

    void infectionsAtLocations() {
        auto realThis = static_cast<SimulationType*>(this);
        for (const auto& loc : realThis->locations) {}
    }
};