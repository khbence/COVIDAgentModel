#pragma once
#include "datatypes.h"
#include "agentsList.h"
#include "location.h"
#include "timeHandler.h"
#include "customExceptions.h"
#include "globalStates.h"
#include "statistics.h"

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
    AgentListType* agents = AgentListType::getInstance();
    unsigned timeStep = 10;

    friend class MovementPolicy<Simulation>;
    friend class InfectionPolicy<Simulation>;
    // We can make it to a singleton later, but who knows
    void refreshAndPrintStatistics() {
        auto init = locations.begin()->refreshAndGetStatistic();
        auto result =
            std::accumulate(locations.begin() + 1, locations.end(), init, [](auto& sum, auto& loc) {
                const auto& stat = loc.refreshAndGetStatistic();
                for (unsigned i = 0; i < sum.size(); ++i) { sum[i] += stat[i]; }
                return sum;
            });
        for (auto val : result) { std::cout << val << ", "; }
        std::cout << '\n';
    }

public:
    void addLocation(PositionType p, TypeOfLocation t) { locations.push_back(LocationType(p, t)); }
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
        PPState_t::printHeader();
        while (simTime < endOfSimulation) {
            if (simTime.isMidnight()) {
                MovementPolicy<Simulation>::planLocations();
                auto& ppstates = agents->PPValues;
                auto& agentMeta = agents->agentMetaData;
                //Update states
                for_each(make_zip_iterator(make_tuple(ppstates.begin(), agentMeta.begin())),
                        make_zip_iterator(make_tuple(ppstates.end(), agentMeta.end())),
                        [](auto tup){
                            auto &ppstate = get<0>(tup);
                            auto &meta = get<1>(tup);
                            ppstate.update(meta.getScalingSymptoms());
                        });
                refreshAndPrintStatistics();
            }
            MovementPolicy<Simulation>::movement();
            InfectionPolicy<Simulation>::infectionsAtLocations(timeStep);
            ++simTime;
        }
    }
};