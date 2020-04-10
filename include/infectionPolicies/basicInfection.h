#pragma once
#include "basicStats.h"

template<typename SimulationType>
class BasicInfection {
public:
    using StatisticType = BasicStats;

    void infectionsAtLocations() {
        auto realThis = static_cast<SimulationType*>(this);
        for (const auto& loc : realThis->locations) {}
    }
};