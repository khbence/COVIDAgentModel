#pragma once
#include "globalStates.h"
#include <cassert>

class PPStateSIRBasic {
    states::SIRD state = states::SIRD::S;

public:
    PPStateSIRBasic() = default;
    explicit PPStateSIRBasic(states::SIRD s) : state(s) {}
    void update(/*elapsed time step + agent meta*/);
    void gotInfected();
    [[nodiscard]] states::SIRD getSIRD() const;
    [[nodiscard]] states::WBStates getWBState() const;
};