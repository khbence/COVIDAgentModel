#pragma once
#include "datatypes.h"
#include "agentsList.h"
#include "location.h"
#include "timeHandler.h"
#include "customExceptions.h"
#include "globalStates.h"
#include "statistics.h"
#include "timing.h"
#include "util.h"

template<typename PositionType,
    typename TypeOfLocation,
    typename PPState,
    typename AgentMeta,
    template<typename>
    typename MovementPolicy,
    template<typename>
    typename InfectionPolicy>
class Simulation
    : private MovementPolicy<Simulation<PositionType,
          TypeOfLocation,
          PPState,
          AgentMeta,
          MovementPolicy,
          InfectionPolicy>>
    , InfectionPolicy<Simulation<PositionType,
          TypeOfLocation,
          PPState,
          AgentMeta,
          MovementPolicy,
          InfectionPolicy>> {

public:
    using PPState_t = PPState;
    using AgentMeta_t = AgentMeta;
    using LocationType = Location<Simulation>;
    using PositionType_t = PositionType;
    using TypeOfLocation_t = TypeOfLocation;
    using AgentListType = AgentList<PPState_t, AgentMeta_t, LocationType>;

private:
    std::vector<LocationType> locations;
    thrust::device_vector<unsigned> locationAgentList; //indices of agents sorted by location, and sorted by agent index
    thrust::device_vector<unsigned> locationIdsOfAgents; //indices of locations of the agents sorted by location, and sorted by agent index
    thrust::device_vector<unsigned> locationListOffsets; //into locationAgentList

    AgentListType* agents = AgentListType::getInstance();
    unsigned timeStep = 10;

    Statistic<PPState, Agent<AgentListType>> globalStats;

    friend class MovementPolicy<Simulation>;
    friend class InfectionPolicy<Simulation>;

    void updateAgents() {
        PROFILE_FUNCTION();
        auto& ppstates = agents->PPValues;
        auto& agentMeta = agents->agentMetaData;
        // Update states
        thrust::for_each(
            thrust::make_zip_iterator(thrust::make_tuple(ppstates.begin(), agentMeta.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(ppstates.end(), agentMeta.end())),
            [](auto tup) {
                auto& ppstate = thrust::get<0>(tup);
                auto& meta = thrust::get<1>(tup);
                ppstate.update(meta.getScalingSymptoms());
            });
    }

    void refreshAndPrintStatistics() {
        PROFILE_FUNCTION();
        std::pair<unsigned, unsigned> idxs{locationListOffsets[0],locationListOffsets.back()};
        auto result = globalStats.refreshandGetAfterMidnight(idxs,locationAgentList);
        for (auto val : result) { std::cout << val << ", "; }
        std::cout << '\n';
    }

public:
    void addLocation(PositionType p, TypeOfLocation t) { locations.push_back(LocationType(p, t)); }
    void addAgent(PPState_t state, bool isDiagnosed, unsigned locationID) {
        unsigned idx = agents->addAgent(state, isDiagnosed, locationID);
    }

    //Must be called after all agents have been added
    bool initialization() {
        PROFILE_FUNCTION();
        locationAgentList.resize(agents->location.size());
        locationIdsOfAgents.resize(agents->location.size());
        locationListOffsets.resize(locations.size()+1);
        Util::updatePerLocationAgentLists(agents->location,locationIdsOfAgents,locationAgentList,locationListOffsets);
        try {
            PPState_t::initTransitionMatrix("../inputFiles/transition.json");
        } catch (TransitionInputError& e) {
            std::cerr << e.what();
            return false;
        }
        return true;
    }

    void runSimulation(unsigned timeStep_p, unsigned lengthOfSimulationWeeks) {
        PROFILE_FUNCTION();
        auto& agentList = agents->getAgentsList();
        timeStep = timeStep_p;
        Timehandler simTime(timeStep);
        const Timehandler endOfSimulation(timeStep, lengthOfSimulationWeeks);
        PPState_t::printHeader();
        while (simTime < endOfSimulation) {
            if (simTime.isMidnight()) {
                MovementPolicy<Simulation>::planLocations();
                updateAgents();
                refreshAndPrintStatistics();
            }
            MovementPolicy<Simulation>::movement(simTime, timeStep);
            InfectionPolicy<Simulation>::infectionsAtLocations(timeStep);
            ++simTime;
        }
    }
};