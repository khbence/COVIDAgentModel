#pragma once
#include "globalStates.h"
#include <cassert>

class PPStateSIRBasic {
    states::SIRD state = states::SIRD::S;

public:
    PPStateSIRBasic() = default;
    explicit PPStateSIRBasic(states::SIRD s) : state(s) {}
    void update(/*agent states (counter, Age, anamnesis, condition related parameters, covid condition related parameters)*/);
    void gotInfected();
    [[nodiscard]] states::SIRD getSIRD() const;
    [[nodiscard]] states::WBStates getWBState() const;
};