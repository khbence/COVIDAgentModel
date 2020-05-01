#pragma once
#include "agentsList.h"
#include "location.h"
#include "timeHandler.h"
#include "customExceptions.h"
#include "globalStates.h"

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
    // using LocationType = Location<Simulation, typename
    // InfectionPolicy<Simulation>::StatisticType>;
    using PPState_t = PPState;
    using AgentMeta_t = AgentMeta;
    using StatisticType = typename InfectionPolicy<Simulation>::StatisticType;
    using LocationType = Location<Simulation, StatisticType>;
    using PositionType_t = PositionType;
    using TypeOfLocation_t = TypeOfLocation;
    using AgentListType = AgentList<PPState_t, AgentMeta_t, LocationType>;

private:
    std::vector<LocationType> locations;
    AgentListType* agents = AgentListType::getInstance();
    unsigned timeStep = 10;

    friend class MovementPolicy<Simulation>;
    friend class InfectionPolicy<Simulation>;
    // We can make it to a singleton later, but who knows
public:
    void addLocation(PositionType p, TypeOfLocation t) { locations.emplace_back(p, t); }
    void addAgent(PPState_t state, bool isDiagnosed, unsigned locationID) {
        agents->addAgent(state, isDiagnosed, &locations[locationID]);
    }

    bool initialization() {
        try {
            PPState_t::initTransitionMatrix("../inputFiles/transition.json");
        } catch (TransitionInputError& e) {
            std::cerr << e.what();
            return false;
        }
        return true;
    }

    void runSimulation(unsigned timeStep_p, unsigned lengthOfSimulationWeeks) {
        auto& agentList = agents->getAgentsList();
        timeStep = timeStep_p;
        Timehandler simTime(timeStep);
        const Timehandler endOfSimulation(timeStep, lengthOfSimulationWeeks);
        while (simTime < endOfSimulation) {
            if (simTime.isMidnight()) {
                MovementPolicy<Simulation>::planLocations();
                for (auto& a : agentList) { a.progressDisease(); }
                // simTime.printDay();
            }
            MovementPolicy<Simulation>::movement();
            InfectionPolicy<Simulation>::infectionsAtLocations(timeStep);
            ++simTime;
        }
    }
};