#pragma once
#include "PPStateSIRBasic.h"

template<typename SimulationType>
class BasicProgression {
protected:
    using PPStateType = PPStateSIRBasic;

    void updateDiseaseStates() {
        auto realThis = static_cast<SimulationType*>(this);
        auto& agents = realThis->agents->getAgentsList();
        for (auto& agent : agents) {
            // Here comes the progression code
        }
    }
};