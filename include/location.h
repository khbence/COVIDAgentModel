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
#include "util.h"

// concept
template<typename SimulationType>
class LocationsList {
    using AgentType = Agent<typename SimulationType::AgentListType>;

    using PositionType = typename SimulationType::PositionType_t;
    using TypeOfLocation = typename SimulationType::TypeOfLocation_t;

    Statistic<typename SimulationType::PPState_t, AgentType> globalStats;

    LocationsList() {}

public:
    thrust::device_vector<PositionType> position;
    thrust::device_vector<TypeOfLocation> locType;
    thrust::device_vector<unsigned>
        locationAgentList;// indices of agents sorted by location, and sorted by agent index
    thrust::device_vector<unsigned> locationIdsOfAgents;// indices of locations of the agents sorted
                                                        // by location, and sorted by agent index
    thrust::device_vector<unsigned> locationListOffsets;// into locationAgentList


    [[nodiscard]] static LocationsList* getInstance() {
        static LocationsList instance;
        return &instance;
    }

    void addLocation(PositionType p, TypeOfLocation l) {
        if (position.size() == position.capacity()) {
            locType.reserve(locType.size() * 1.5 + 10);
            position.reserve(position.size() * 1.5 + 10);
        }
        position.push_back(p);
        locType.push_back(l);
    }

    void initialize() {
        auto agents = SimulationType::AgentListType::getInstance();
        locationAgentList.resize(agents->location.size());
        locationIdsOfAgents.resize(agents->location.size());
        locationListOffsets.resize(position.size() + 1);
        Util::updatePerLocationAgentLists(
            agents->location, locationIdsOfAgents, locationAgentList, locationListOffsets);
    }

    /*    std::pair<unsigned,unsigned>& getAgents() { return agents; }

        void addAgent(unsigned a) {
            stat.refreshStatisticNewAgent(a);
        }

        void removeAgent(unsigned idx) {
            stat.refreshStatisticRemoveAgent(idx);
        }*/

    // TODO optimise randoms for performance
    static void infectAgents(thrust::device_vector<double>& infectionRatioAtLocations,
        thrust::device_vector<unsigned>& agentLocations) {
        PROFILE_FUNCTION();
        auto& ppstates = SimulationType::AgentListType::getInstance()->PPValues;
        // DEBUG unsigned count1 = thrust::count_if(ppstates.begin(),ppstates.end(), [](auto
        // &ppstate) {return ppstate.getSIRD() == states::SIRD::I;}); DESC: for (int i = 0; i <
        // number_of_agents; i++) {ppstate = ppstates[i]; infectionRatio =
        // infectionRatioAtLocations[agentLocations[i]];...}
        thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(ppstates.begin(),
                             thrust::make_permutation_iterator(
                                 infectionRatioAtLocations.begin(), agentLocations.begin()))),
            thrust::make_zip_iterator(thrust::make_tuple(ppstates.end(),
                thrust::make_permutation_iterator(
                    infectionRatioAtLocations.begin(), agentLocations.end()))),
            [] HD (thrust::tuple<typename SimulationType::PPState_t &,double &> tuple) {
                auto& ppstate = thrust::get<0>(tuple);
                double& infectionRatio = thrust::get<1>(tuple);
                if (ppstate.getSIRD() == states::SIRD::S
                    && RandomGenerator::randomUnit() < infectionRatio) {
                    ppstate.gotInfected();
                }
            });
        // DEBUG unsigned count2 = thrust::count_if(ppstates.begin(),ppstates.end(), [](auto
        // &ppstate) {return ppstate.getSIRD() == states::SIRD::I;}); DEBUG std::cout << count1 <<  "
        // " << count2 << std::endl;
    }

    const auto& refreshAndGetStatistic() {
        std::pair<unsigned, unsigned> agents{ locationListOffsets[0], locationListOffsets.back() };
        return globalStats.refreshandGetAfterMidnight(agents, locationAgentList);
    }
};