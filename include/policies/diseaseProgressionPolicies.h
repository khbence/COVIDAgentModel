#pragma once
#include "PPStateTypes.h"

template<typename SimulationType>
class ExtendedProgression {
protected:
    using PPStateType = PPStateSIRextended;

    void updateDiseaseStates() {
        auto realThis = static_cast<SimulationType*>(this);
        auto& agents = realThis->agents->getAgentsList();
        int stats[4] = {0};
        int subStates[6] = {0};
        for (auto& agent : agents) { 
            agent.progressDisease(/*additional scaling factor = 1.0*/);
            stats[(int)agent.getSIRDState()]++;
            if (agent.getSIRDState() == states::SIRD::I) {
                subStates[(int)agent.getPPState().getSubState()]++;
            }
        }
        std::cout << stats[0] << ", " << stats[1] << ", " << stats[2] << ", " << stats[3];
        std::cout << ", " << subStates[0] << ", " << subStates[1] << ", " << subStates[2] 
                  << ", " << subStates[3] << ", " << subStates[4] << ", " << subStates[5] << std::endl;
    }
};