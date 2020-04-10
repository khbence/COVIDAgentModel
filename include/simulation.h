#pragma once
#include "agentsList.h"
#include "location.h"
#include "timeHandler.h"

template<typename PPState,
    typename PositionType,
    typename TypeOfLocation,
    template<typename>
    typename MovementPolicy,
    template<typename>
    typename InfectionPolicy>
class Simulation
    : private MovementPolicy<Simulation<PPState, PositionType, TypeOfLocation, MovementPolicy, InfectionPolicy>>
    , InfectionPolicy<Simulation<PPState, PositionType, TypeOfLocation, MovementPolicy, InfectionPolicy>> {

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
    AgentList<PPState, LocationType>* agentList() { return agents; }
    // TODO add addLocation function instead
    std::vector<LocationType>& locationList() { return locations; }

    void runSimulation(unsigned lengthOfSimulationWeeks) {
        MovementPolicy<Simulation>::movement();
        InfectionPolicy<Simulation>::infectionsAtLocations();
        std::cout << Timehandler<10>(lengthOfSimulationWeeks) << '\n';
        /*
        if(daySwitch) planLocation()
        movement();
        locationUpdate(); //what is the infection rate there + infectAgents(); infectionAtLocation()
        updateConditions();
        */
    }
};