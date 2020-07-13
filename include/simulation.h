#pragma once
#include "datatypes.h"
#include "agentsList.h"
#include "locationList.h"
#include "timeHandler.h"
#include "customExceptions.h"
#include "globalStates.h"
#include "statistics.h"
#include "timing.h"
#include "util.h"
#include <cxxopts.hpp>
#include "dataProvider.h"

template<typename PositionType,
    typename TypeOfLocation,
    typename PPState,
    typename AgentMeta,
    template<typename>
    typename MovementPolicy,
    template<typename>
    typename InfectionPolicy>
class Simulation
    : private MovementPolicy<Simulation<PositionType, TypeOfLocation, PPState, AgentMeta, MovementPolicy, InfectionPolicy>>
    , InfectionPolicy<Simulation<PositionType, TypeOfLocation, PPState, AgentMeta, MovementPolicy, InfectionPolicy>> {

public:
    using PPState_t = PPState;
    using AgentMeta_t = AgentMeta;
    using LocationType = LocationsList<Simulation>;
    using PositionType_t = PositionType;
    using TypeOfLocation_t = TypeOfLocation;
    using AgentListType = AgentList<PPState_t, AgentMeta_t, LocationType>;

    // private:
    AgentListType* agents = AgentListType::getInstance();
    LocationType* locs = LocationType::getInstance();
    unsigned timeStep;
    unsigned lengthOfSimulationWeeks;
    bool succesfullyInitialized = true;
    bool singleLocation;

    friend class MovementPolicy<Simulation>;
    friend class InfectionPolicy<Simulation>;

    static void addProgramParameters(cxxopts::Options& options) {
        InfectionPolicy<Simulation>::addProgramParameters(options);
        MovementPolicy<Simulation>::addProgramParameters(options);
    }

    void updateAgents(Timehandler &simTime) {
        PROFILE_FUNCTION();
        auto& ppstates = agents->PPValues;
        auto& agentStats = agents->agentStats;
        auto& agentMeta = agents->agentMetaData;
        unsigned timestamp = simTime.getTimestamp();
        // Update states
        thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(ppstates.begin(), agentMeta.begin(), agentStats.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(ppstates.end(), agentMeta.end(), agentStats.end())),
            [timestamp] HD(thrust::tuple<PPState&, AgentMeta&, AgentStats&> tup) {
                auto& ppstate = thrust::get<0>(tup);
                auto& meta = thrust::get<1>(tup);
                auto agentStat = thrust::get<2>(tup);
                ppstate.update(meta.getScalingSymptoms(), agentStat, timestamp);
            });
    }

    void refreshAndPrintStatistics() {
        PROFILE_FUNCTION();
        auto result = locs->refreshAndGetStatistic();
        for (auto val : result) { std::cout << val << "\t"; }
        std::cout << '\n';
    }

public:
    explicit Simulation(const cxxopts::ParseResult& result)
        : timeStep(result["deltat"].as<decltype(timeStep)>()),
          lengthOfSimulationWeeks(result["weeks"].as<decltype(lengthOfSimulationWeeks)>()),
          singleLocation(result["numlocs"].as<int>() == 1) {
        PROFILE_FUNCTION();
        InfectionPolicy<Simulation>::initializeArgs(result);
        MovementPolicy<Simulation>::initializeArgs(result);
        DataProvider data{ result };
        try {
            std::string header = PPState_t::initTransitionMatrix(data.acquireProgressionMatrix());
            agents->initAgentMeta(data.acquireParameters());
            locs->initLocationTypes(data.acquireLocationTypes());
            MovementPolicy<Simulation>::init(data.acquireLocationTypes());
            auto locationMapping = locs->initLocations(data.acquireLocations());
            auto agentTypeMapping = agents->initAgentTypes(data.acquireAgentTypes());
            agents->initAgents(data.acquireAgents(), locationMapping, agentTypeMapping, data.getAgentTypeLocTypes());
            RandomGenerator::resize(agents->PPValues.size());
            std::cout << header << '\n';
        } catch (const CustomErrors& e) {
            std::cerr << e.what();
            succesfullyInitialized = false;
        }
        locs->initialize();
    }

    void runSimulation() {
        if (!succesfullyInitialized) { return; }
        PROFILE_FUNCTION();
        Timehandler simTime(timeStep);
        const Timehandler endOfSimulation(timeStep, lengthOfSimulationWeeks);
        refreshAndPrintStatistics();
        while (simTime < endOfSimulation) {
            if (simTime.isMidnight()) {
                MovementPolicy<Simulation>::planLocations();
                updateAgents(simTime);
                refreshAndPrintStatistics();
            }
            MovementPolicy<Simulation>::movement(simTime, timeStep);
            if (singleLocation) {
                InfectionPolicy<Simulation>::infectionSingleLocation(simTime, timeStep);
            } else {
                InfectionPolicy<Simulation>::infectionsAtLocations(simTime, timeStep);
            }
            ++simTime;
        }
    }
};