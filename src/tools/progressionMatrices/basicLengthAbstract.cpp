#include "basicLengthAbstract.h"
#include "smallTools.h"

double BasicLengthAbstract::LengthOfState::expectedLength(double p, unsigned max) const {
    if(p < 0.0) {
        return 0.0;
    }
    double sumExpected = 0.0;
    double sumChance = 0.0;
    for(unsigned i = 0; i < max; ++i) {
        double chance = std::pow(1-p, i)*p; 
        sumExpected += (i+1)*chance;
        sumChance += chance;
    }
    return sumExpected*(1.0 / sumChance);
}

double BasicLengthAbstract::LengthOfState::calculateModifiedP() const {
    double n = static_cast<double>(avgLength);
    double pNew = 1.0 / n;
    auto objFunction = [](double pCurrent) {
        return expectedValue(maxLength, pCurrent) - n;
    };
    try {
        pNew = SecantMethod(objFunction, pNew, pNew - 0.01, 0.0001);
    }
    catch(const std::runtime_error& e) {
        std::cerr << e.what() << ' original p value will be used instead \n';
    }
    // To check if an extremely small p value 
    if(pNew * 10.0 < (1.0/n)) {
        pNew = 1.0/n;
    }
    return pNew;
}

BasicLengthAbstract::LengthOfState::LengthOfState(int avgLength_p, int maxLength_p)
    : avgLength(avgLength_p), maxLength(maxLength_p), p(1.0 / static_cast<double>(avgLength_p + 1)) {
    if (maxLength == -1) { maxLength = std::numeric_limits<decltype(maxLength)>::max(); }
}

// Note: [0, maxLength), because the 0 will run for a day, so the maxLength
// would run for maxLength+1 days
[[nodiscard]] HD int BasicLengthAbstract::LengthOfState::calculateDays() const {
    if (avgLength == -1) { return -1; }
    int days = RandomGenerator::geometric(p);
    while (maxLength < days) { days = RandomGenerator::geometric(p); }
    return days;
}

BasicLengthAbstract::BasicLengthAbstract(std::size_t n)
    : numStates(n), lengths((LengthOfState*)malloc(sizeof(LengthOfState) * n)) {}

HD int BasicLengthAbstract::calculateJustDays(unsigned state) const {
    return lengths[state].calculateDays();
}