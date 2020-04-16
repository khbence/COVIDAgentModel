#pragma once
#include "PPStateSIRBasic.h"

template<typename SimulationType>
class BasicProgression {
protected:
    using PPStateType = PPStateSIRBasic;

    void updateDiseaseStates() {}
};