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
#include "programParameters.h"
#include <string>
#include "locationTypesFormat.h"
#include <map>
#include <unordered_map>
#include "locationsFormat.h"
#include "customExceptions.h"

// concept
template<typename SimulationType>
class LocationsList {
    using AgentType = Agent<typename SimulationType::AgentListType>;

    using PositionType = typename SimulationType::PositionType_t;
    using TypeOfLocation = typename SimulationType::TypeOfLocation_t;

    Statistic<typename SimulationType::PPState_t, AgentType> globalStats;

    // For the runtime performance, it would be better, that the IDs of the locations would be the
    // same as their indexes, but we can not ensure it in the input file, so I create this mapping,
    // that will be used by the agents when I fill them up. Use it only during initialization ID
    // from files -> index in vectors
    std::unordered_map<unsigned, unsigned> IDMapping;

    LocationsList() = default;

    void reserve(std::size_t s) {
        position.reserve(s);
        locType.reserve(s);
        areas.reserve(s);
        states.reserve(s);
    }

public:
    // the following vectors are the input data for locations in separated vectors
    thrust::device_vector<TypeOfLocation> locType;
    thrust::device_vector<PositionType> position;
    thrust::device_vector<unsigned> areas;
    thrust::device_vector<bool> states;// Closed/open or ON/OFF


    // indices of agents sorted by location, and sorted by agent index
    thrust::device_vector<unsigned> locationAgentList;
    // indices of locations of the agents sorted
    // by location, and sorted by agent index
    thrust::device_vector<unsigned> locationIdsOfAgents;
    // into locationAgentList
    thrust::device_vector<unsigned> locationListOffsets;

    std::map<unsigned, std::string> generalLocationTypes;

    [[nodiscard]] static LocationsList* getInstance() {
        static LocationsList instance;
        return &instance;
    }

    void initLocationTypes(const std::string& locationTypeFile) {
        auto input = DECODE_JSON_FILE(locationTypeFile, parser::LocationTypes);
        for (auto& type : input.types) {
            generalLocationTypes.emplace(std::make_pair(type.ID, std::move(type.name)));
        }
    }

    void initLocations(const std::string& locationFile) {
        auto input = DECODE_JSON_FILE(locationFile, parser::Locations);
        reserve(input.places.size());
        unsigned idx = 0;
        for (const auto& loc : input.places) {
            IDMapping.emplace(loc.ID, idx++);
            locType.push_back(loc.type);
            position.push_back(PositionType{ loc.coordinates[0], loc.coordinates[1] });
            areas.push_back(loc.area);
            // Transform to upper case, to make it case insensitive
            std::string tmp = loc.state;
            std::for_each(tmp.begin(), tmp.end(), [](char c) { return std::toupper(c); });
            if (tmp == "ON" || tmp == "OPEN") {
                states.push_back(true);
            } else if (tmp == "OFF" || tmp == "CLOSED") {
                states.push_back(false);
            } else {
                throw IOLocations::WrongState(loc.state);
            }
        }
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
            [](auto tuple) {
                auto& ppstate = thrust::get<0>(tuple);
                double& infectionRatio = thrust::get<1>(tuple);
                if (ppstate.getSIRD() == states::SIRD::S
                    && RandomGenerator::randomUnit() < infectionRatio) {
                    ppstate.gotInfected();
                }
            });
        // DEBUG unsigned count2 = thrust::count_if(ppstates.begin(),ppstates.end(), [](auto
        // &ppstate) {return ppstate.getSIRD() == states::SIRD::I;}); DEBUG std::cout << count1 << "
        // " << count2 << std::endl;
    }

    const auto& refreshAndGetStatistic() {
        std::pair<unsigned, unsigned> agents{ locationListOffsets[0], locationListOffsets.back() };
        return globalStats.refreshandGetAfterMidnight(agents, locationAgentList);
    }
};