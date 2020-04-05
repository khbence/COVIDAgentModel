#pragma once
#include "globalStates.h"
#include "markovChain.h"

class PPStateSIRextended {
    states::SIRD state = states::SIRD::S;
    char counter = 0; //I1, I2, I3 ... R1, R2
    static MarkovChain<9> mc; //specific for this, but uses only indexes not the enum type

public:
    void update(/*elapsed time step?*/);
    void gotInfected();
    states::SIRD getSIRD() const;
    states::WBStates getWBState() const;
};