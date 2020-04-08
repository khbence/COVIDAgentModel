#pragma once
#include "globalStates.h"

class PPStateSIRBasic {
    states::SIRD state = states::SIRD::S;

public:
	PPStateSIRBasic()=default;
	PPStateSIRBasic(states::SIRD s) : state(s) {}
    void update(/*elapsed time step + agent meta*/);
    void gotInfected();
    [[nodiscard]] states::SIRD getSIRD() const;
    [[nodiscard]] states::WBStates getWBState() const;
};