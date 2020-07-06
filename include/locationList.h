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
#include <string>
#include "locationTypesFormat.h"
#include <map>
#include "locationsFormat.h"
#include "customExceptions.h"

template<typename SimulationType>
class LocationsList {
    using AgentType = Agent<typename SimulationType::AgentListType>;

    using PositionType = typename SimulationType::PositionType_t;
    using TypeOfLocation = typename SimulationType::TypeOfLocation_t;

    Statistic<typename SimulationType::PPState_t, AgentType> globalStats;

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

    void initLocationTypes(const parser::LocationTypes& inputData) {
        for (auto& type : inputData.types) { generalLocationTypes.emplace(std::make_pair(type.ID, std::move(type.name))); }
    }

    [[nodiscard]] std::map<std::string, unsigned> initLocations(const parser::Locations& inputData) {
        // For the runtime performance, it would be better, that the IDs of the locations would be
        // the same as their indexes, but we can not ensure it in the input file, so I create this
        // mapping, that will be used by the agents when I fill them up. Use it only during
        // initialization ID from files -> index in vectors
        std::map<std::string, unsigned> IDMapping{};

        thrust::host_vector<TypeOfLocation> locType_h;
        thrust::host_vector<PositionType> position_h;
        thrust::host_vector<unsigned> areas_h;
        thrust::host_vector<bool> states_h;

        reserve(inputData.places.size());
        unsigned idx = 0;
        for (const auto& loc : inputData.places) {
            IDMapping.emplace(loc.ID, idx++);
            locType_h.push_back(loc.type);
            position_h.push_back(PositionType{ loc.coordinates[0], loc.coordinates[1] });
            areas_h.push_back(loc.area);
            // Transform to upper case, to make it case insensitive
            std::string tmp = loc.state;
            std::for_each(tmp.begin(), tmp.end(), [](char c) { return std::toupper(c); });
            if (tmp == "ON" || tmp == "OPEN") {
                states_h.push_back(true);
            } else if (tmp == "OFF" || tmp == "CLOSED") {
                states_h.push_back(false);
            } else {
                throw IOLocations::WrongState(loc.state);
            }
        }

        locType = locType_h;
        position = position_h;
        areas = areas_h;
        states = states_h;

        return IDMapping;
    }

    void initialize() {
        auto agents = SimulationType::AgentListType::getInstance();
        locationAgentList.resize(agents->location.size());
        locationIdsOfAgents.resize(agents->location.size());
        locationListOffsets.resize(position.size() + 1);
        Util::updatePerLocationAgentLists(agents->location, locationIdsOfAgents, locationAgentList, locationListOffsets);
    }

    // TODO optimise randoms for performance
    static void infectAgents(thrust::device_vector<double>& infectionRatioAtLocations, thrust::device_vector<unsigned>& agentLocations) {
        PROFILE_FUNCTION();
        auto& ppstates = SimulationType::AgentListType::getInstance()->PPValues;
        // DEBUG unsigned count1 = thrust::count_if(ppstates.begin(),ppstates.end(), [](auto
        // &ppstate) {return ppstate.getSIRD() == states::SIRD::I;}); DESC: for (int i = 0; i <
        // number_of_agents; i++) {ppstate = ppstates[i]; infectionRatio =
        // infectionRatioAtLocations[agentLocations[i]];...}
        thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(
                             ppstates.begin(), thrust::make_permutation_iterator(infectionRatioAtLocations.begin(), agentLocations.begin()))),
            thrust::make_zip_iterator(
                thrust::make_tuple(ppstates.end(), thrust::make_permutation_iterator(infectionRatioAtLocations.begin(), agentLocations.end()))),
            [] HD(thrust::tuple<typename SimulationType::PPState_t&, double&> tuple) {
                auto& ppstate = thrust::get<0>(tuple);
                double& infectionRatio = thrust::get<1>(tuple);
                if (ppstate.isSusceptible() && RandomGenerator::randomUnit() < infectionRatio) { ppstate.gotInfected(); }
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