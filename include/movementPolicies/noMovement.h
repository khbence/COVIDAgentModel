#include <iostream>

template<typename SimulationType>
class NoMovement {
protected:
    void movement() { std::cout << static_cast<SimulationType*>(this)->locations.size() << '\n'; }
};