#pragma once
#include "agent.h"

class BasicStats {
private:
    unsigned sick = 0;
    unsigned infected = 0;
    unsigned healthy = 0;

public:
    [[nodiscard]] unsigned getSick() const { return sick; }
    [[nodiscard]] unsigned getNewlyInfected() const { return infected; }
    [[nodiscard]] unsigned getHealthy() const { return healthy; }
    void setSick(unsigned sick_p) { sick = sick_p; }
    void setNewlyInfected(unsigned infected_p) { infected = infected_p; }
    void setHealthy(unsigned healthy_p) { healthy = healthy_p; }

    template<typename AgentType>
    void newAgent(const AgentType a) {}
};