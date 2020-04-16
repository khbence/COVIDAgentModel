#pragma once
#include "agentsList.h"
#include "location.h"
#include "basicStats.h"
#include "timeHandler.h"

template<typename PositionType,
    typename TypeOfLocation,
    template<typename>
    typename MovementPolicy,
    template<typename>
    typename InfectionPolicy,
    template<typename>
    typename ProgressionPolicy>
class Simulation
    : private MovementPolicy<Simulation<PositionType,
          TypeOfLocation,
          MovementPolicy,
          InfectionPolicy,
          ProgressionPolicy>>
    , InfectionPolicy<Simulation<PositionType,
          TypeOfLocation,
          MovementPolicy,
          InfectionPolicy,
          ProgressionPolicy>>
    , ProgressionPolicy<Simulation<PositionType,
          TypeOfLocation,
          MovementPolicy,
          InfectionPolicy,
          ProgressionPolicy>> {

public:
    // using LocationType = Location<Simulation, typename
    // InfectionPolicy<Simulation>::StatisticType>;
    using PPState_t = typename ProgressionPolicy<Simulation>::PPStateType;
    using StatisticType = BasicStats;
    using LocationType = Location<Simulation, StatisticType>;
    using PositionType_t = PositionType;
    using TypeOfLocation_t = TypeOfLocation;
    using AgentListType = AgentList<PPState_t, LocationType>;

private:
    std::vector<LocationType> locations;
    AgentListType* agents = AgentList<PPState_t, LocationType>::getInstance();

    friend class MovementPolicy<Simulation>;
    friend class InfectionPolicy<Simulation>;
    friend class ProgressionPolicy<Simulation>;
    // We can make it to a singleton later, but who knows
public:
    void addLocation(PositionType p, TypeOfLocation t) { locations.emplace_back(p, t); }
    void addAgent(PPState_t state, bool isDiagnosed, unsigned locationID) {
        agents->addAgent(state, isDiagnosed, &locations[locationID]);
    }

    template<unsigned timeStep>
    void runSimulation(unsigned lengthOfSimulationWeeks) {
        Timehandler<timeStep> simTime;
        const Timehandler<timeStep> endOfSimulation(lengthOfSimulationWeeks);
        while (simTime < endOfSimulation) {
            if (simTime.isMidnight()) {
                MovementPolicy<Simulation>::planLocations();//+disease progession
                ProgressionPolicy<Simulation>::updateDiseaseStates();
                simTime.printDay();
            }
            MovementPolicy<Simulation>::movement();
            InfectionPolicy<Simulation>::infectionsAtLocations();
            ++simTime;
        }
    }
};