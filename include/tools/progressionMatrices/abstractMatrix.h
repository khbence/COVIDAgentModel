#pragma once
#include "datatypes.h"
#include <tuple>

class AbstractMatrix {
public:
    [[nodiscard]] HD thrust::pair<unsigned, int> calculateNextState(unsigned currentState, float scalingSymptons) const {return thrust::pair<unsigned, int>(0,0);};
    [[nodiscard]] HD int calculateJustDays(unsigned state) const {return 0;};
};