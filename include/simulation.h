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
#include "programParameters.h"

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

private:
    AgentListType* agents = AgentListType::getInstance();
    LocationType* locs = LocationType::getInstance();
    unsigned timeStep;
    unsigned lengthOfSimulationWeeks;

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
        auto result = locs->refreshAndGetStatistic();
        for (auto val : result) { std::cout << val << ", "; }
        std::cout << '\n';
    }

public:
    explicit Simulation(const ProgramParameters& parameters)
        : timeStep(parameters.timeStep), lengthOfSimulationWeeks(parameters.weeks) {
        try {
            PPState_t::initTransitionMatrix(parameters.progression);
        } catch (TransitionInputError& e) { std::cerr << e.what(); }
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

    void runSimulation() {
        PROFILE_FUNCTION();
        auto& agentList = agents->getAgentsList();
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