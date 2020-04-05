#pragma once
#include <array>

template<unsigned N>
class MarkovChain {
    std::array<std::array<double, N>, N> behaviour; //I know continous representation is better, later

public:
    MarkovChain(std::array<std::array<double, N>, N> behaviour_p);
    unsigned nextState(unsigned current) const;
};