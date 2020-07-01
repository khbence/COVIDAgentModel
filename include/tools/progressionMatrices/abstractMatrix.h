#pragma once
#include "datatypes.h"
#include <tuple>

class AbstractMatrix {
public:
    [[nodiscard]] virtual HD thrust::pair<unsigned, int> calculateNextState(unsigned currentState, float scalingSymptons) const = 0;
    [[nodiscard]] virtual HD int calculateJustDays(unsigned state) const = 0;
};