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
#include "timing.h"

// concept
template<typename SimulationType>
class Location {
    using AgentType = Agent<typename SimulationType::AgentListType>;

    typename SimulationType::PositionType_t position;
    typename SimulationType::TypeOfLocation_t locType;
    std::pair<unsigned,unsigned> agents;
    //Statistic<typename SimulationType::PPState_t, AgentType> stat;

public:
    Location(decltype(position) p, decltype(locType) t) : position(p), locType(t) {}

    std::pair<unsigned,unsigned>& getAgents() { return agents; }

/*    void addAgent(unsigned a) {
        stat.refreshStatisticNewAgent(a);
    }

    void removeAgent(unsigned idx) {
        stat.refreshStatisticRemoveAgent(idx);
    }*/

    // TODO optimise randoms for performance
    static void infectAgents(thrust::device_vector<double> &infectionRatioAtLocations,
                             thrust::device_vector<unsigned> &agentLocations) {
        PROFILE_FUNCTION();
        auto& ppstates = SimulationType::AgentListType::getInstance()->PPValues;
        //DEBUG unsigned count1 = thrust::count_if(ppstates.begin(),ppstates.end(), [](auto &ppstate) {return ppstate.getSIRD() == states::SIRD::I;});
        //DESC: for (int i = 0; i < number_of_agents; i++) {ppstate = ppstates[i]; infectionRatio = infectionRatioAtLocations[agentLocations[i]];...}
        thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(ppstates.begin(),
                                                   thrust::make_permutation_iterator(infectionRatioAtLocations.begin(), agentLocations.begin()))),
                         thrust::make_zip_iterator(thrust::make_tuple(ppstates.end(),
                                                   thrust::make_permutation_iterator(infectionRatioAtLocations.begin(), agentLocations.end()))),
                         [](auto tuple) {
                            auto& ppstate = thrust::get<0>(tuple);
                            double& infectionRatio = thrust::get<1>(tuple);
                            if (ppstate.getSIRD() == states::SIRD::S && RandomGenerator::randomUnit() < infectionRatio) {
                                ppstate.gotInfected();
                            }
                         });
        //DEBUG unsigned count2 = thrust::count_if(ppstates.begin(),ppstates.end(), [](auto &ppstate) {return ppstate.getSIRD() == states::SIRD::I;});
        //DEBUG std::cout << count1 <<  " " << count2 << std::endl;
    }

    //const auto& refreshAndGetStatistic(thrust::device_vector<unsigned> &locationAgentList) {
    //    //TODO: this should only be called after agents was updated 
    //    return stat.refreshandGetAfterMidnight(agents, locationAgentList);
    //}
};