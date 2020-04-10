#pragma once
#include "basicStats.h"
#include <iostream>

template<typename SimulationType>
class BasicInfection {
    double getInfectionRatio(/*to be defined*/) { return 0.1; }

public:
    using StatisticType = BasicStats;

    void infectionsAtLocations() {
        auto realThis = static_cast<SimulationType*>(this);
        for (auto& loc : realThis->locations) {
            // we can ask for other stuffs too
            loc.infectAgents(getInfectionRatio());
            loc.cleanUp();
            unsigned infected = loc.getInfected();
            unsigned healthy = loc.getHealthy();
            std::cout << "Healthy: " << healthy << " - Infected: " << infected << " Date: ";
        }
    }
};