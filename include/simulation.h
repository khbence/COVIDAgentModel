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
    : public MovementPolicy<Simulation<PPState, PositionType, TypeOfLocation, MovementPolicy, InfectionPolicy>>
    , public InfectionPolicy<Simulation<PPState, PositionType, TypeOfLocation, MovementPolicy, InfectionPolicy>> {

    using LocationType = Location<PositionType, TypeOfLocation, typename InfectionPolicy<Simulation>::StatisticType>;

    std::vector<LocationType> locations;
    AgentList<PPState, LocationType>* agents = AgentList<PPState, LocationType>::getInstance();

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