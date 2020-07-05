#pragma once
#include "randomGenerator.h"

class BasicLengthAbstract {
protected:
    class LengthOfState {
        int avgLength;
        int maxLength;
        double p;

    public:
        LengthOfState() = default;
        LengthOfState(int avgLength_p, int maxLength_p);
        [[nodiscard]] HD int calculateDays() const;
    };

    BasicLengthAbstract(std::size_t n);

public:
    LengthOfState* lengths;
    unsigned numStates;

    [[nodiscard]] HD int calculateJustDays(unsigned state) const;
};