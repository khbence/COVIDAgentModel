#pragma once

template<typename SimulationType>
class BasicProgression {
protected:
    void updateDiseaseStates() {
        auto realThis = static_cast<SimulationType*>(this);
        auto& agents = realThis->agents->getAgentsList();
        for (auto &agent : agents) {
            //Here comes the progression code
        }
    }
};