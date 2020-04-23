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
        for (auto& agent : agents) { 
            agent.progressDisease(/*additional scaling factor = 1.0*/);
            stats[(int)agent.getSIRDState()]++;
        }
        std::cout << "S: " << stats[0] << " I: " << stats[1] << " R: " << stats[2] << " D: " << stats[3] << std::endl;
    }
};