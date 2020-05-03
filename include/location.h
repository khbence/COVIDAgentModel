#pragma once
#include <vector>
#include "globalStates.h"
#include "agent.h"
#include <cmath>
#include <algorithm>
#include <random>
#include "randomGenerator.h"
#include "statistics.h"
#include "datatypes.h"

// concept
template<typename SimulationType>
class Location {
    using AgentType = Agent<typename SimulationType::AgentListType>;

    typename SimulationType::PositionType_t position;
    typename SimulationType::TypeOfLocation_t locType;
    thrust::device_vector<unsigned> agents;
    Statistic<typename SimulationType::PPState_t, AgentType> stat;

public:
    Location(decltype(position) p, decltype(locType) t) : position(p), locType(t) {}

    thrust::device_vector<unsigned>& getAgents() { return agents; }

    void addAgent(unsigned a) {
        agents.push_back(a);
        stat.refreshStatisticNewAgent(a);
    }

    void removeAgent(unsigned idx) {
        // agents.back().swap(agents[idx]);
        thrust::swap(agents.back(), agents[idx]);
        stat.refreshStatisticRemoveAgent(agents.back());
        agents.pop_back();
    }

    // TODO optimise randoms for performance
    void infectAgents(double ratio) {
        auto& ppstates = SimulationType::AgentListType::getInstance()->PPValues;
        thrust::device_vector<float> rnds(agents.size());
        rnds = RandomGenerator::fillUnitf(agents.size());
        //auto &rnds_host 
        //thrust::copy(rnds_host.begin(), rnds_host.end(), rnds.begin());

        int newInfections = thrust::transform_reduce(
            thrust::make_zip_iterator(thrust::make_tuple(rnds.begin(),
                thrust::make_permutation_iterator(ppstates.begin(), agents.begin()))),
            thrust::make_zip_iterator(thrust::make_tuple(rnds.end(),
                thrust::make_permutation_iterator(ppstates.begin(), agents.end()))),
            [=](auto& i) {
                auto &rnd = thrust::get<0>(i);
                auto &a = thrust::get<1>(i);
                if (a.getSIRD() == states::SIRD::S && rnd < ratio) {
                    a.gotInfected();
                    return 1;
                }
                return 0;
            },
            0,
            thrust::plus<int>());
        //if (newInfections) std::cout << newInfections << std::endl;
    }

    const auto& refreshAndGetStatistic() { return stat.refreshandGetAfterMidnight(agents); }
};