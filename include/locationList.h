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
#include "timeHandler.h"
#include <cxxopts.hpp>

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
        capacity.reserve(s);
        states.reserve(s);
        quarantineUntil.reserve(s);
    }

public:
    // the following vectors are the input data for locations in separated
    // vectors
    thrust::device_vector<TypeOfLocation> locType;
    thrust::device_vector<PositionType> position;
    thrust::device_vector<double> infectiousness;
    thrust::device_vector<unsigned> areas;
    thrust::device_vector<unsigned> capacity;
    thrust::device_vector<bool> states;// Closed/open or ON/OFF
    thrust::device_vector<unsigned> quarantineUntil;


    // indices of agents sorted by location, and sorted by agent index
    thrust::device_vector<unsigned> locationAgentList;
    // indices of locations of the agents sorted
    // by location, and sorted by agent index
    thrust::device_vector<unsigned> locationIdsOfAgents;
    // into locationAgentList
    thrust::device_vector<unsigned> locationListOffsets;

    std::map<unsigned, std::string> generalLocationTypes;

    unsigned tracked;

    [[nodiscard]] static LocationsList* getInstance() {
        static LocationsList instance;
        return &instance;
    }

    void initLocationTypes(const parser::LocationTypes& inputData) {
        for (auto& type : inputData.types) {
            generalLocationTypes.emplace(std::make_pair(type.ID, std::move(type.name)));
        }
        unsigned cemeteryTypeID = generalLocationTypes.rbegin()->first + 1;
        generalLocationTypes.emplace(std::make_pair(cemeteryTypeID, "cemetery"));
    }

    void initializeArgs(const cxxopts::ParseResult& result) {
        try {
            tracked = result["trace"].as<unsigned>();
        } catch (std::exception& e) { tracked = std::numeric_limits<unsigned>::max(); }
    }

    [[nodiscard]] std::pair<unsigned, std::map<std::string, unsigned>> initLocations(
        const parser::Locations& inputData) {
        // For the runtime performance, it would be better, that the IDs of the
        // locations would be the same as their indexes, but we can not ensure
        // it in the input file, so I create this mapping, that will be used by
        // the agents when I fill them up. Use it only during initialization ID
        // from files -> index in vectors
        std::map<std::string, unsigned> IDMapping{};

        thrust::host_vector<TypeOfLocation> locType_h;
        thrust::host_vector<PositionType> position_h;
        thrust::host_vector<double> infectiousness_h;
        thrust::host_vector<unsigned> areas_h;
        thrust::host_vector<bool> states_h;
        thrust::host_vector<unsigned> capacity_h;
        thrust::host_vector<unsigned> quarantineUntil_h;
        auto s = inputData.places.size() + 1;//+1 because of the cemetery
        locType_h.reserve(s);
        position_h.reserve(s);
        infectiousness_h.reserve(s);
        areas_h.reserve(s);
        states_h.reserve(s);
        capacity_h.reserve(s);
        quarantineUntil_h.reserve(s);

        reserve(s);
        unsigned idx = 0;
        for (const auto& loc : inputData.places) {
            IDMapping.emplace(loc.ID, idx++);
            locType_h.push_back(loc.type);
            position_h.push_back(PositionType{ loc.coordinates[0], loc.coordinates[1] });
            infectiousness_h.push_back(loc.infectious);
            capacity_h.push_back(loc.capacity);
            areas_h.push_back(loc.area);
            quarantineUntil_h.push_back(0);
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

        // adding cemetery
        locType_h.push_back(generalLocationTypes.rbegin()->first);
        position_h.push_back(PositionType{ 0, 0 });
        infectiousness_h.push_back(0.0);
        areas_h.push_back(std::numeric_limits<unsigned>::max());
        states_h.push_back(true);
        capacity_h.push_back(std::numeric_limits<unsigned>::max());

        locType = locType_h;
        position = position_h;
        infectiousness = infectiousness_h;
        areas = areas_h;
        states = states_h;
        capacity = capacity_h;
        quarantineUntil = quarantineUntil_h;

        return std::make_pair(locType.size() - 1, IDMapping);
    }

    void initialize() {
        auto agents = SimulationType::AgentListType::getInstance();
        locationAgentList.resize(agents->location.size());
        locationIdsOfAgents.resize(agents->location.size());
        locationListOffsets.resize(position.size() + 1);
        Util::updatePerLocationAgentLists(
            agents->location, locationIdsOfAgents, locationAgentList, locationListOffsets);
    }

    // TODO optimise randoms for performance
    static void infectAgents(thrust::device_vector<double>& infectionRatioAtLocations,
        thrust::device_vector<unsigned>& agentLocations,
        thrust::device_vector<bool>& infectionAtLocation,
        bool flagInfectionsAtLocation,
        Timehandler& simTime) {
        PROFILE_FUNCTION();
        auto& ppstates = SimulationType::AgentListType::getInstance()->PPValues;
        auto& agentStats = SimulationType::AgentListType::getInstance()->agentStats;
        unsigned timestamp = simTime.getTimestamp();
        unsigned tracked2 = getInstance()->tracked;
        // DEBUG unsigned count1 =
        // thrust::count_if(ppstates.begin(),ppstates.end(), [](auto &ppstate)
        // {return ppstate.getSIRD() == states::SIRD::I;}); DESC: for (int i =
        // 0; i < number_of_agents; i++) {ppstate = ppstates[i]; infectionRatio
        // = infectionRatioAtLocations[agentLocations[i]];...}
        thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(ppstates.begin(),
                             thrust::make_permutation_iterator(
                                 infectionRatioAtLocations.begin(), agentLocations.begin()),
                             thrust::make_permutation_iterator(
                                 getInstance()->locType.begin(), agentLocations.begin()),
                             agentStats.begin(),
                             agentLocations.begin(),
                             thrust::make_counting_iterator<unsigned>(0),
                             thrust::make_permutation_iterator(
                                 infectionAtLocation.begin(), agentLocations.begin()))),
            thrust::make_zip_iterator(thrust::make_tuple(ppstates.end(),
                thrust::make_permutation_iterator(
                    infectionRatioAtLocations.begin(), agentLocations.end()),
                thrust::make_permutation_iterator(
                    getInstance()->locType.begin(), agentLocations.end()),
                agentStats.end(),
                agentLocations.end(),
                thrust::make_counting_iterator<unsigned>(0) + ppstates.size(),
                thrust::make_permutation_iterator(
                                 infectionAtLocation.begin(), agentLocations.end()))),
            [timestamp, tracked2,flagInfectionsAtLocation] HD(thrust::tuple<typename SimulationType::PPState_t&,
                double&,
                TypeOfLocation&,
                AgentStats&,
                unsigned&,
                unsigned,
                bool&> tuple) {
                auto& ppstate = thrust::get<0>(tuple);
                double& infectionRatio = thrust::get<1>(tuple);
                TypeOfLocation& locType = thrust::get<2>(tuple);
                auto& agentStat = thrust::get<3>(tuple);
                unsigned& agentLocation = thrust::get<4>(tuple);
                unsigned agentID = thrust::get<5>(tuple);
                bool& infectionAtLocation = thrust::get<6>(tuple);
                if (ppstate.isSusceptible() && RandomGenerator::randomUnit() < infectionRatio) {
                    ppstate.gotInfected();
                    agentStat.infectedTimestamp = timestamp;
                    agentStat.infectedLocation = agentLocation;
                    agentStat.worstState = ppstate.getStateIdx();
                    agentStat.worstStateTimestamp = timestamp;
                    if (flagInfectionsAtLocation)
                        infectionAtLocation = true;
                    if (agentID == tracked2) {
                        printf(
                            "Agent %d got infected at location %d of type %d at timestamp %d\n",
                            agentID,
                            agentLocation,
                            locType,
                            timestamp);
                    }
                }
            });
        // DEBUG unsigned count2 =
        // thrust::count_if(ppstates.begin(),ppstates.end(), [](auto &ppstate)
        // {return ppstate.getSIRD() == states::SIRD::I;}); DEBUG std::cout <<
        // count1 << " " << count2 << std::endl;
    }

    const auto& refreshAndGetStatistic() {
        std::pair<unsigned, unsigned> agents{ locationListOffsets[0], locationListOffsets.back() };
        return globalStats.refreshandGetAfterMidnight(agents, locationAgentList);
    }
};
