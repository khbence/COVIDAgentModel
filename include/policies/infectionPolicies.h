#pragma once
#include "location.h"
#include "basicStats.h"
#include <iostream>

template<class SimulationType>
class BasicInfection {
    double virulency = 0.0;

    template<typename LocationType>
    double getInfectionRatio(LocationType& loc) {

        auto& agents = loc.getAgents();
        auto realThis = static_cast<SimulationType*>(this);
        unsigned numInfectedAgentsPresent = std::count_if(agents.begin(),
            agents.end(),
            [](auto agent) { return agent.getSIRDState() == states::SIRD::I; });
        unsigned total = agents.size();
        if (numInfectedAgentsPresent == 0) return 0;
        double densityOfInfected = double(numInfectedAgentsPresent) / 1000.0;
        double p = 0.35 - virulency;
        double k = p / 5.0;
        double y = 1.0 / (1.0 + exp((p - densityOfInfected) / k));
        unsigned timeStepCopy = static_cast<SimulationType*>(this)->timeStep;
        return y / 144.0;// Has to be scaled with time I think
    }

protected:
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