#pragma once
#include "agent.h"
#include "globalStates.h"

class BasicStats {
private:
    unsigned infected = 0;
    // do we need it? or just set the infecteds directly
    unsigned newlyInfected = 0;
    unsigned healthy = 0;

public:
    [[nodiscard]] unsigned getInfected() const { return infected; }
    [[nodiscard]] unsigned getNewlyInfected() const { return newlyInfected; }
    [[nodiscard]] unsigned getHealthy() const { return healthy; }
    void setInfected(unsigned infected_p) { infected = infected_p; }
    void setNewlyInfected(unsigned newlyInfected_p) { newlyInfected = newlyInfected_p; }
    void setHealthy(unsigned healthy_p) { healthy = healthy_p; }

    // obligatory for every stat type
    template<typename AgentListType>
    void refreshStatisticNewAgent(const Agent<AgentListType>& a) {
        auto state = a.getSIRDState();
        switch (state) {
        case states::SIRD::I:
            ++infected;
            break;
        case states::SIRD::S:
            ++healthy;
            break;
        default:
            break;
        }
    }

    void cleanUp() {
        infected += newlyInfected;
        healthy -= newlyInfected;
        newlyInfected = 0;
    }
};