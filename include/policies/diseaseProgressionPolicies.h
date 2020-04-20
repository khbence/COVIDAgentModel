#pragma once
#include "PPStateTypes.h"

template<typename SimulationType>
class ExtendedProgression {
protected:
    using PPStateType = PPStateSIRextended;

    void updateDiseaseStates() {
        auto realThis = static_cast<SimulationType*>(this);
        auto& agents = realThis->agents->getAgentsList();
        for (auto& agent : agents) { agent.progressDisease(/*additional scaling factor = 1.0*/); }
    }
};