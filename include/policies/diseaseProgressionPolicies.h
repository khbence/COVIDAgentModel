#pragma once
#include "PPStateTypes.h"

template<typename SimulationType>
class ExtendedProgression {
protected:
    using PPStateType = PPStateSIRextended;

    void updateDiseaseStates() {
        auto realThis = static_cast<SimulationType*>(this);
        auto& ppstates = realThis->agents->PPValues;
        int stats[4] = {0};
        int subStates[6] = {0};
        for_each(ppstates.begin(), ppstates.end(),
        [&] (auto &ppstate) { 
            //agent.progressDisease(/*additional scaling factor = 1.0*/);
            stats[(int)ppstate.getSIRD()]++;
            if (ppstate.getSIRD() == states::SIRD::I) {
                subStates[(int)ppstate.getSubState()]++;
            }
        });
        std::cout << stats[0] << ", " << stats[1] << ", " << stats[2] << ", " << stats[3];
        std::cout << ", " << subStates[0] << ", " << subStates[1] << ", " << subStates[2] 
                  << ", " << subStates[3] << ", " << subStates[4] << ", " << subStates[5] << std::endl;
    }
};