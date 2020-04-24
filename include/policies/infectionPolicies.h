#pragma once
#include "location.h"
#include "basicStats.h"
#include <iostream>

template<class SimulationType>
class BasicInfection {
    double virulency = 0.0;

    template<typename LocationType>
    double getInfectionRatio(LocationType& loc, unsigned timeStep) {

        auto& agents = loc.getAgents();
        auto realThis = static_cast<SimulationType*>(this);
        unsigned numInfectedAgentsPresent = std::count_if(agents.begin(),
            agents.end(),
            [](auto agent) { return agent.getSIRDState() == states::SIRD::I; });
        unsigned total = agents.size();
        if (numInfectedAgentsPresent == 0) return 0;
        double densityOfInfected = double(numInfectedAgentsPresent) / (agents.size()*1.2);
        double p = 0.35 - virulency;
        double k = p / 5.0;
        double y = 1.0 / (1.0 + exp((p - densityOfInfected) / k));
        unsigned timeStepCopy = static_cast<SimulationType*>(this)->timeStep;
        return y / (60.0*24.0/(double)timeStep);
    }

protected:
    using StatisticType = BasicStats;

    void infectionsAtLocations(unsigned timeStep) {
        auto realThis = static_cast<SimulationType*>(this);
        for (auto& loc : realThis->locations) {
            loc.infectAgents(getInfectionRatio(loc, timeStep));
            loc.cleanUp();
            unsigned infected = loc.getInfected();
            unsigned healthy = loc.getHealthy();
            // std::cout << "Healthy: " << healthy << " - Infected: " << infected << " Date: ";
        }
    }
};