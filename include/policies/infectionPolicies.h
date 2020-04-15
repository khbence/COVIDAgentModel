#pragma once
#include "basicStats.h"
#include <iostream>

template<typename SimulationType>
class BasicInfection {
    double virulency = 0.0;
    double getInfectionRatio(typename SimulationType::LocationType loc) { 
        std::vector<unsigned>& agents = loc.getAgents();
        auto realThis = static_cast<SimulationType*>(this);
        unsigned numInfectedAgentsPresent = std::count_if(agents.begin(), agents.end(), [](unsigned i) {
                return realThis->agents->getPPState(i).getSIRD() ==
    states::SIRD::I;
            });
        unsigned total = agents.size();
        if (numInfectedAgentsPresent == 0) return 0;
        double densityOfInfected = double(numInfectedAgentsPresent)/1.0;
        double p = 0.35 - virulency;
        double k = p/5.0;
        double y = 1.0/(1.0+exp((p-densityOfInfected)/k));
        return y;
    }

public:
    using StatisticType = BasicStats;

    void infectionsAtLocations() {
        auto realThis = static_cast<SimulationType*>(this);
        for (auto& loc : realThis->locations) {
            loc.infectAgents(getInfectionRatio(loc));
            loc.cleanUp();
            unsigned infected = loc.getInfected();
            unsigned healthy = loc.getHealthy();
            // std::cout << "Healthy: " << healthy << " - Infected: " << infected << " Date: ";
        }
    }
};