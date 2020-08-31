#pragma once
#include <limits>
#include <iostream>

class AgentStats {
public:
    unsigned infectedTimestamp = std::numeric_limits<unsigned>::max();
    unsigned infectedLocation = 0;
    unsigned worstStateTimestamp = 0;
    unsigned worstStateEndTimestamp = 0;
    unsigned diagnosedTimestamp = 0;
    unsigned quarantinedTimestamp = 0;
    unsigned quarantinedUntilTimestamp = 0;
    char worstState = 0;
    friend std::ostream& operator<<(std::ostream& os, const AgentStats& s);
};