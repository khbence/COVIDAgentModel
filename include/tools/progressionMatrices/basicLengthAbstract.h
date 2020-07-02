#pragma once
#include "abstractMatrix.h"
#include "randomGenerator.h"

class BasicLengthAbstract : public AbstractMatrix {
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

public:
    LengthOfState* lengths;
    unsigned numStates;

    [[nodiscard]] int HD calculateJustDays(unsigned state) const override;
};