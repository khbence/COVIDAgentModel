#pragma once
#include "globalStates.h"

class PPStateSIRextended {
    states::SIRD state = states::SIRD::S;
    char counter = 0; //I1, I2, I3 ... R1, R2

public:
    void update(/*elapsed time step + agent meta*/);
    void gotInfected();
    [[nodiscard]] states::SIRD getSIRD() const;
    [[nodiscard]] states::WBStates getWBState() const;
};