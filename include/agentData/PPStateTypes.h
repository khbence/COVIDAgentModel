#pragma once
#include "globalStates.h"
#include "transitionMatrix.h"

class PPStateSIRAbstract {
protected:
    states::SIRD state;

public:
    static unsigned getNumberOfStates() {return 0;};

    static void initTransitionMatrix(const std::string& inputFile) {}

    explicit PPStateSIRAbstract(states::SIRD s);
    virtual void update(float scalingSymptons) = 0;
    virtual void gotInfected();
    [[nodiscard]] states::SIRD getSIRD() const;
    [[nodiscard]] states::WBStates getWBState() const;
    virtual char getStateIdx() const = 0;
};

class PPStateSIRBasic : public PPStateSIRAbstract {
public:
    static unsigned getNumberOfStates() {return 4;};
    PPStateSIRBasic();
    explicit PPStateSIRBasic(states::SIRD s);
    void update(float scalingSymptons) override;
};

class PPStateSIRextended : public PPStateSIRAbstract {
    char subState = 0;// I1, I2, I3 ... R1, R2
    char idx = 0;

    // -1 it will remain in that state until something special event happens, like got infected
    // -2 has to be calculated during update
    int daysBeforeNextState = -1;

private:
    SingleBadTransitionMatrix& getTransition();

    void applyNewIdx();

public:
    static void printHeader();

    static unsigned getNumberOfStates();
    PPStateSIRextended();
    explicit PPStateSIRextended(states::SIRD s);
    explicit PPStateSIRextended(char idx_p);
    void gotInfected() override;
    [[nodiscard]] char getSubState() { return subState; }
    static void initTransitionMatrix(const std::string& inputFile);
    void update(float scalingSymptons) override;
    [[nodiscard]] char getStateIdx() const override;
};
