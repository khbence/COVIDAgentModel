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
#include <cxxopts.hpp>

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
    using LocationType = LocationsList<Simulation>;
    using PositionType_t = PositionType;
    using TypeOfLocation_t = TypeOfLocation;
    using AgentListType = AgentList<PPState_t, AgentMeta_t, LocationType>;

//private:
    AgentListType* agents = AgentListType::getInstance();
    LocationType* locs = LocationType::getInstance();
    unsigned timeStep = 10;

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
            [] __host__ __device__ (thrust::tuple<PPState &, AgentMeta &> tup) {
                auto& ppstate = thrust::get<0>(tup);
                auto& meta = thrust::get<1>(tup);
                ppstate.update(meta.getScalingSymptoms());
            });
    }

    void refreshAndPrintStatistics() {
        PROFILE_FUNCTION();
        auto result = locs->refreshAndGetStatistic();
        for (auto val : result) { std::cout << val << ", "; }
        std::cout << '\n';
    }

public:
    Simulation(cxxopts::Options& options)
        : InfectionPolicy<Simulation<PositionType,
            TypeOfLocation,
            PPState,
            AgentMeta,
            MovementPolicy,
            InfectionPolicy>>(options) {}
    void initialize_args(cxxopts::ParseResult& result) {
        InfectionPolicy<Simulation<PositionType,
            TypeOfLocation,
            PPState,
            AgentMeta,
            MovementPolicy,
            InfectionPolicy>>::initialize_args(result);
        try {
            PPState_t::initTransitionMatrix("../inputFiles/transition.json");
        } catch (TransitionInputError& e) {
            std::cerr << e.what();
            // return false;
        }
    }

    void addLocation(PositionType p, TypeOfLocation t) { locs->addLocation(p, t); }
    void addAgent(PPState_t state, bool isDiagnosed, unsigned locationID) {
        unsigned idx = agents->addAgent(state, isDiagnosed, locationID);
    }

    // Must be called after all agents have been added
    bool initialization() {
        PROFILE_FUNCTION();
        locs->initialize();
        return true;
    }

    void runSimulation(unsigned timeStep_p, unsigned lengthOfSimulationWeeks) {
        PROFILE_FUNCTION();
        auto& agentList = agents->getAgentsList();
        timeStep = timeStep_p;
        Timehandler simTime(timeStep);
        const Timehandler endOfSimulation(timeStep, lengthOfSimulationWeeks);
        PPState_t::printHeader();
        refreshAndPrintStatistics();
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