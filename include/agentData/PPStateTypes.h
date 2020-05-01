#pragma once
#include "globalStates.h"
#include "transitionMatrix.h"

class PPStateSIRAbstract {
protected:
    states::SIRD state;

public:
    static void initTransitionMatrix(const std::string& inputFile) {}

    explicit PPStateSIRAbstract(states::SIRD s);
    virtual void update(float scalingSymptons) = 0;
    virtual void gotInfected();
    [[nodiscard]] states::SIRD getSIRD() const;
    [[nodiscard]] states::WBStates getWBState() const;
};

class PPStateSIRBasic : public PPStateSIRAbstract {
public:
    PPStateSIRBasic();
    explicit PPStateSIRBasic(states::SIRD s);
    void update(float scalingSymptons) override;
};

class PPStateSIRextended : public PPStateSIRAbstract {
    char subState = 0;// I1, I2, I3 ... R1, R2
    char idx = 0;

    // future
    int daysBeforeNextState = -1;

    static constexpr unsigned numberOfStates = 1 + 6 + 3 + 1;// S + I + R + D
    static constexpr std::array<unsigned, 5> startingIdx{ 0, 1, 7, 10, 11 };// to convert from idx to state
    static inline SingleBadTransitionMatrix<numberOfStates> transition;

    void applyNewIdx();

public:
    PPStateSIRextended();
    explicit PPStateSIRextended(states::SIRD s);
    explicit PPStateSIRextended(char idx_p);
    void gotInfected() override;
    char getSubState() {return subState;}
    char getIdx() {return idx;}
    static void initTransitionMatrix(const std::string& inputFile) {
        transition = decltype(transition)(inputFile);
    }
    void update(float scalingSymptons) override;
};