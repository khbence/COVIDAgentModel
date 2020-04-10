#pragma once
#include "agentsList.h"
#include "location.h"
#include "timeHandler.h"

template<typename PPState,
    typename PositionType,
    typename TypeOfLocation,
    template<typename>
    typename PlanningPolicy,
    template<typename>
    typename MovementPolicy,
    template<typename>
    typename InfectionPolicy>
class Simulation
    : private PlanningPolicy<Simulation<PPState,
          PositionType,
          TypeOfLocation,
          PlanningPolicy,
          MovementPolicy,
          InfectionPolicy>>
    , MovementPolicy<Simulation<PPState,
          PositionType,
          TypeOfLocation,
          PlanningPolicy,
          MovementPolicy,
          InfectionPolicy>>
    , InfectionPolicy<Simulation<PPState,
          PositionType,
          TypeOfLocation,
          PlanningPolicy,
          MovementPolicy,
          InfectionPolicy>> {

public:
    using LocationType = Location<Simulation, typename InfectionPolicy<Simulation>::StatisticType>;
    using PPState_t = PPState;
    using PositionType_t = PositionType;
    using TypeOfLocation_t = TypeOfLocation;
    using AgentListType = AgentList<PPState, LocationType>;

private:
    std::vector<LocationType> locations;
    AgentListType* agents = AgentList<PPState, LocationType>::getInstance();

    friend class MovementPolicy<Simulation>;
    friend class InfectionPolicy<Simulation>;
    // We can make it to a singleton later, but who knows
public:
    void addLocation(PositionType p, TypeOfLocation t) { locations.emplace_back(p, t); }
    void addAgent(PPState state, bool isDiagnosed, unsigned locationID) {
        agents->addAgent(state, isDiagnosed, &locations[locationID]);
    }

    template<unsigned timeStep>
    void runSimulation(unsigned lengthOfSimulationWeeks) {
        Timehandler<timeStep> simTime;
        const Timehandler<timeStep> endOfSimulation(lengthOfSimulationWeeks);
        while (simTime < endOfSimulation) {
            if (simTime.isMidnight()) { PlanningPolicy<Simulation>::planLocation(); }
            MovementPolicy<Simulation>::movement();
            InfectionPolicy<Simulation>::infectionsAtLocations();
            ++simTime;
        }
        /*
        if(daySwitch) planLocation()
        movement();
        locationUpdate(); //what is the infection rate there + infectAgents(); infectionAtLocation()
        updateConditions();
        */
    }
};