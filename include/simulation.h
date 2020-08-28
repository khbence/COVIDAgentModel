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
    typename InfectionPolicy,
    template<typename>
    typename TestingPolicy>
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

    // private:
    AgentListType* agents = AgentListType::getInstance();
    LocationType* locs = LocationType::getInstance();
    unsigned timeStep;
    unsigned lengthOfSimulationWeeks;
    bool succesfullyInitialized = true;
    std::string outAgentStat;
    int enableSuddenDeath = 1;

    friend class MovementPolicy<Simulation>;
    friend class InfectionPolicy<Simulation>;
    friend class TestingPolicy<Simulation>;

    static void addProgramParameters(cxxopts::Options& options) {
        options.add_options()("suddenDeath",
            "Enable (1) or disable (2) non-COVID sudden death ",
            cxxopts::value<int>()->default_value("1"));
        InfectionPolicy<Simulation>::addProgramParameters(options);
        MovementPolicy<Simulation>::addProgramParameters(options);
        TestingPolicy<Simulation>::addProgramParameters(options);
    }

    void suddenDeath(Timehandler& simTime) {
        PROFILE_FUNCTION();
        auto& ppstates = agents->PPValues;
        auto& agentStats = agents->agentStats;
        auto& agentMeta = agents->agentMetaData;
        unsigned timestamp = simTime.getTimestamp();
        unsigned tracked = locs->tracked;
        thrust::for_each(
            thrust::make_zip_iterator(thrust::make_tuple(ppstates.begin(),
                agentMeta.begin(),
                agentStats.begin(),
                thrust::make_counting_iterator<unsigned>(0))),
            thrust::make_zip_iterator(thrust::make_tuple(ppstates.end(),
                agentMeta.end(),
                agentStats.end(),
                thrust::make_counting_iterator<unsigned>(0) + ppstates.size())),
            [timestamp, tracked] HD(
                thrust::tuple<PPState&, AgentMeta&, AgentStats&, unsigned>
                    tup) {
                auto& ppstate = thrust::get<0>(tup);
                auto& meta = thrust::get<1>(tup);
                auto& agentStat = thrust::get<2>(tup);
                unsigned agentID = thrust::get<3>(tup);
                uint8_t age = meta.getAge();
                bool sex = meta.getSex();
                double probability = 0.0;
                // If already dead, or in hospital, return
                if (ppstate.getWBState() == states::WBStates::D
                    || ppstate.getWBState() == states::WBStates::S)
                    return;
                if (age < 5) {
                    probability =
                        sex ? 0.000138636206246181 : 0.0000742144645456176;
                } else if (age < 15) {
                    probability =
                        sex ? 0.000138636206246181 : 0.0000742144645456176;
                } else if (age < 30) {
                    probability =
                        sex ? 0.000138636206246181 : 0.0000742144645456176;
                } else if (age < 60) {
                    probability =
                        sex ? 0.00140203198030386 : 0.000435754402861572;
                } else if (age < 70) {
                    probability =
                        sex ? 0.00731842791450156 : 0.00323823204036677;
                } else if (age < 80) {
                    probability =
                        sex ? 0.0127014602887186 : 0.00719076352220422;
                } else {
                    probability = sex ? 0.0384859186350238 : 0.0355438319317952;
                }
                probability /= 100.0;
                if (RandomGenerator::randomReal(1.0) < probability) {
                    agentStat.worstState = ppstate.die();
                    agentStat.worstStateTimestamp = timestamp;
                    // printf("Agent %d died of sudden death, %d, timestamp
                    // %d\n", agentID, (int)agentStat.worstState,timestamp);
                    if (agentID == tracked) {
                        printf("Agent %d died of sudden death, timestamp %d\n",
                            tracked,
                            timestamp);
                    }
                }
            });
    }

    void updateAgents(Timehandler& simTime) {
        PROFILE_FUNCTION();
        auto& ppstates = agents->PPValues;
        auto& agentStats = agents->agentStats;
        auto& agentMeta = agents->agentMetaData;
        auto& diagnosed = agents->diagnosed;
        unsigned timestamp = simTime.getTimestamp();
        unsigned tracked = locs->tracked;
        // Update states
        thrust::for_each(
            thrust::make_zip_iterator(thrust::make_tuple(ppstates.begin(),
                agentMeta.begin(),
                agentStats.begin(),
                diagnosed.begin(),
                thrust::make_counting_iterator<unsigned>(0))),
            thrust::make_zip_iterator(thrust::make_tuple(ppstates.end(),
                agentMeta.end(),
                agentStats.end(),
                diagnosed.end(),
                thrust::make_counting_iterator<unsigned>(0) + ppstates.size())),
            [timestamp, tracked] HD(thrust::
                    tuple<PPState&, AgentMeta&, AgentStats&, bool&, unsigned>
                        tup) {
                auto& ppstate = thrust::get<0>(tup);
                auto& meta = thrust::get<1>(tup);
                auto& agentStat = thrust::get<2>(tup);
                auto& diagnosed = thrust::get<3>(tup);
                unsigned agentID = thrust::get<4>(tup);
                bool recovered = ppstate.update(meta.getScalingSymptoms(),
                    agentStat,
                    timestamp,
                    agentID,
                    tracked);
                if (recovered) diagnosed = false;
            });
    }

    void refreshAndPrintStatistics() {
        PROFILE_FUNCTION();
        auto result = locs->refreshAndGetStatistic();
        for (auto val : result) { std::cout << val << "\t"; }
        auto tests = TestingPolicy<Simulation>::getStats();
        std::cout << thrust::get<0>(tests) << "\t" << thrust::get<1>(tests)
                  << "\t" << thrust::get<2>(tests) << "\t";
        std::cout << '\n';
    }

public:
    explicit Simulation(const cxxopts::ParseResult& result)
        : timeStep(result["deltat"].as<decltype(timeStep)>()),
          lengthOfSimulationWeeks(
              result["weeks"].as<decltype(lengthOfSimulationWeeks)>()) {
        PROFILE_FUNCTION();
        outAgentStat = result["outAgentStat"].as<std::string>();
        InfectionPolicy<Simulation>::initializeArgs(result);
        MovementPolicy<Simulation>::initializeArgs(result);
        TestingPolicy<Simulation>::initializeArgs(result);
        enableSuddenDeath = result["suddenDeath"].as<int>();
        DataProvider data{ result };
        try {
            std::string header = PPState_t::initTransitionMatrix(
                data.acquireProgressionMatrices());
            agents->initAgentMeta(data.acquireParameters());
            locs->initLocationTypes(data.acquireLocationTypes());
            auto tmp = locs->initLocations(data.acquireLocations());
            auto cemeteryID = tmp.first;
            auto locationMapping = tmp.second;
            locs->initializeArgs(result);
            MovementPolicy<Simulation>::init(
                data.acquireLocationTypes(), cemeteryID);
            auto agentTypeMapping =
                agents->initAgentTypes(data.acquireAgentTypes());
            agents->initAgents(data.acquireAgents(),
                locationMapping,
                agentTypeMapping,
                data.getAgentTypeLocTypes(),
                data.acquireProgressionMatrices());
            RandomGenerator::resize(agents->PPValues.size());
            std::cout << header << "\tT\tP1\tP2" << '\n';
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
                if (simTime.getTimestamp() > 0)
                    TestingPolicy<Simulation>::performTests(simTime, timeStep);
                if (simTime.getTimestamp() > 0)
                    updateAgents(simTime);// No disease progression at launch
                if (enableSuddenDeath) suddenDeath(simTime);
                refreshAndPrintStatistics();
            }
            MovementPolicy<Simulation>::movement(simTime, timeStep);
            InfectionPolicy<Simulation>::infectionsAtLocations(
                simTime, timeStep);
            ++simTime;
        }
        // thrust::copy(agents->agentStats.begin(), agents->agentStats.end(),
        // std::ostream_iterator<AgentStats>(std::cout, ""));
        agents->printAgentStatJSON(outAgentStat);
    }
};