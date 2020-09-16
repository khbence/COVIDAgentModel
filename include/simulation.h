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
          InfectionPolicy,
          TestingPolicy>>
    , InfectionPolicy<Simulation<PositionType,
          TypeOfLocation,
          PPState,
          AgentMeta,
          MovementPolicy,
          InfectionPolicy,
          TestingPolicy>>
    , TestingPolicy<Simulation<PositionType,
          TypeOfLocation,
          PPState,
          AgentMeta,
          MovementPolicy,
          InfectionPolicy,
          TestingPolicy>> {

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
    int enableOtherDisease = 1;

    friend class MovementPolicy<Simulation>;
    friend class InfectionPolicy<Simulation>;
    friend class TestingPolicy<Simulation>;

    static void addProgramParameters(cxxopts::Options& options) {
        options.add_options()("otherDisease",
            "Enable (1) or disable (2) non-COVID related hospitalization and sudden death ",
            cxxopts::value<int>()->default_value("1"));
        InfectionPolicy<Simulation>::addProgramParameters(options);
        MovementPolicy<Simulation>::addProgramParameters(options);
        TestingPolicy<Simulation>::addProgramParameters(options);
    }

    void otherDisease(Timehandler& simTime, unsigned timeStep) {
        PROFILE_FUNCTION();
        auto& ppstates = agents->PPValues;
        auto& agentStats = agents->agentStats;
        auto& agentMeta = agents->agentMetaData;
        unsigned timestamp = simTime.getTimestamp();
        unsigned tracked = locs->tracked;
        thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(ppstates.begin(),
                             agentMeta.begin(),
                             agentStats.begin(),
                             thrust::make_counting_iterator<unsigned>(0))),
            thrust::make_zip_iterator(thrust::make_tuple(ppstates.end(),
                agentMeta.end(),
                agentStats.end(),
                thrust::make_counting_iterator<unsigned>(0) + ppstates.size())),
            [timestamp, tracked, timeStep] HD(
                thrust::tuple<PPState&, AgentMeta&, AgentStats&, unsigned> tup) {
                auto& ppstate = thrust::get<0>(tup);
                auto& meta = thrust::get<1>(tup);
                auto& agentStat = thrust::get<2>(tup);
                unsigned agentID = thrust::get<3>(tup);

                //
                // non-COVID hospitalization - ended, see if dies or lives
                //
                if (timestamp>0 && agentStat.hospitalizedUntilTimestamp == timestamp) {
                    if (RandomGenerator::randomReal(1.0) < 0.031628835) {
                        agentStat.worstState = ppstate.die(false); //not COVID-related
                        agentStat.worstStateTimestamp = timestamp;
                        //printf("Agent %d died at the end of hospital stay %d\n", agentID, timestamp);
                        if (agentID == tracked) {
                            printf("Agent %d died at the end of hospital stay %d\n", tracked, timestamp);
                        }
                        return;
                    } else {
                        //printf("Agent %d recovered at the end of hospital stay %d\n", agentID, timestamp);
                        if (agentID == tracked) {
                            printf("Agent %d recovered at the end of hospital stay %d\n", tracked, timestamp);
                        }
                    }
                }

                //
                // Sudden death
                //
                uint8_t age = meta.getAge();
                bool sex = meta.getSex();
                double probability = 0.0;
                //If already dead, or in hospital (due to COVID or non-COVID), return
                if (ppstate.getWBState() == states::WBStates::D || 
                    ppstate.getWBState() == states::WBStates::S ||
                    timestamp < agentStat.hospitalizedUntilTimestamp) return;
                if (age < 5) {
                    probability = sex ? 3.79825E-07 : 2.03327E-07;
                } else if (age < 15) {
                    probability = sex ? 3.79825E-07 : 2.03327E-07;
                } else if (age < 30) {
                    probability = sex ? 3.79825E-07 : 2.03327E-07;
                } else if (age < 60) {
                    probability = sex ? 3.84118E-06 : 1.19385E-06;
                } else if (age < 70) {
                    probability = sex ? 2.00505E-05 : 8.87187E-06;
                } else if (age < 80) {
                    probability = sex ? 3.47985E-05 : 1.97007E-05;
                } else {
                    probability = sex ? 0.000105441 : 9.73804E-05;
                }
                probability /= 100.0;
                if (RandomGenerator::randomReal(1.0) < probability) {
                    agentStat.worstState = ppstate.die(false); //not COVID-related
                    agentStat.worstStateTimestamp = timestamp;
                    // printf("Agent %d died of sudden death, %d, timestamp
                    // %d\n", agentID, (int)agentStat.worstState,timestamp);
                    if (agentID == tracked) {
                        printf("Agent %d died of sudden death, timestamp %d\n", tracked, timestamp);
                    }
                    return;
                }

                //
                // Random hospitalization
                //
                probability = 0.000888138;
                if (RandomGenerator::randomReal(1.0) < probability) {
                    //Got hospitalized
                    //Length;
                    unsigned avgLength = 5.55; //In days
                    double p = 1.0 / (double) avgLength;
                    unsigned length = RandomGenerator::geometric(p);
                    if (length == 0) length = 1; // At least one day
                    agentStat.hospitalizedTimestamp = timestamp;
                    agentStat.hospitalizedUntilTimestamp = timestamp + length*24*60/timeStep;
                    //printf("Agent %d hospitalized for non-COVID disease, timestamp %d-%d\n", agentID, timestamp, agentStat.hospitalizedUntilTimestamp);
                    if (agentID == tracked) {
                        printf("Agent %d hospitalized for non-COVID disease, timestamp %d-%d\n", agentID, timestamp, agentStat.hospitalizedUntilTimestamp);
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
        thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(ppstates.begin(),
                             agentMeta.begin(),
                             agentStats.begin(),
                             diagnosed.begin(),
                             thrust::make_counting_iterator<unsigned>(0))),
            thrust::make_zip_iterator(thrust::make_tuple(ppstates.end(),
                agentMeta.end(),
                agentStats.end(),
                diagnosed.end(),
                thrust::make_counting_iterator<unsigned>(0) + ppstates.size())),
            [timestamp, tracked] HD(
                thrust::tuple<PPState&, AgentMeta&, AgentStats&, bool&, unsigned> tup) {
                auto& ppstate = thrust::get<0>(tup);
                auto& meta = thrust::get<1>(tup);
                auto& agentStat = thrust::get<2>(tup);
                auto& diagnosed = thrust::get<3>(tup);
                unsigned agentID = thrust::get<4>(tup);
                bool recovered = ppstate.update(
                    meta.getScalingSymptoms(), agentStat, timestamp, agentID, tracked);
                if (recovered) diagnosed = false;
            });
    }

    void refreshAndPrintStatistics(Timehandler& simTime) {
        PROFILE_FUNCTION();
        //COVID
        auto result = locs->refreshAndGetStatistic();
        for (auto val : result) { std::cout << val << "\t"; }
        //non-COVID hospitalization
        auto& ppstates = agents->PPValues;
        auto& diagnosed = agents->diagnosed;
        auto& agentStats = agents->agentStats;
        unsigned timestamp = simTime.getTimestamp();
        unsigned hospitalized = 
            thrust::count_if(thrust::make_zip_iterator(thrust::make_tuple(ppstates.begin(), agentStats.begin(), diagnosed.begin())),
                             thrust::make_zip_iterator(thrust::make_tuple(ppstates.end(), agentStats.end(), diagnosed.end())),
                         [timestamp] HD (thrust::tuple<PPState, AgentStats, bool> tup) {
                             auto ppstate = thrust::get<0>(tup);
                             auto agentStat = thrust::get<1>(tup);
                             auto diagnosed = thrust::get<2>(tup);
                             if (ppstate.getWBState() != states::WBStates::D &&  //avoid double-counting with COVID
                                 ppstate.getWBState() != states::WBStates::S &&
                                 diagnosed == false &&
                                 timestamp < agentStat.hospitalizedUntilTimestamp) return true;
                             else return false;
                         });
        std::cout << hospitalized << "\t";
        //Testing
        auto tests = TestingPolicy<Simulation>::getStats();
        std::cout << thrust::get<0>(tests) << "\t" << thrust::get<1>(tests) << "\t"
                  << thrust::get<2>(tests) << "\t";
        //Quarantine stats
        auto quarant = agents->getQuarantineStats(timestamp);
        std::cout << thrust::get<0>(quarant) << "\t" << thrust::get<1>(quarant) << "\t"
                  << thrust::get<2>(quarant) << "\t";
        std::cout << '\n';
    }

public:
    explicit Simulation(const cxxopts::ParseResult& result)
        : timeStep(result["deltat"].as<decltype(timeStep)>()),
          lengthOfSimulationWeeks(result["weeks"].as<decltype(lengthOfSimulationWeeks)>()) {
        PROFILE_FUNCTION();
        outAgentStat = result["outAgentStat"].as<std::string>();
        InfectionPolicy<Simulation>::initializeArgs(result);
        MovementPolicy<Simulation>::initializeArgs(result);
        TestingPolicy<Simulation>::initializeArgs(result);
        enableOtherDisease = result["otherDisease"].as<int>();
        DataProvider data{ result };
        try {
            std::string header = PPState_t::initTransitionMatrix(
                data.acquireProgressionMatrices(), data.acquireProgressionConfig());
            agents->initAgentMeta(data.acquireParameters());
            locs->initLocationTypes(data.acquireLocationTypes());
            auto tmp = locs->initLocations(data.acquireLocations());
            auto cemeteryID = tmp.first;
            auto locationMapping = tmp.second;
            locs->initializeArgs(result);
            MovementPolicy<Simulation>::init(data.acquireLocationTypes(), cemeteryID);
            TestingPolicy<Simulation>::init(data.acquireLocationTypes());
            auto agentTypeMapping = agents->initAgentTypes(data.acquireAgentTypes());
            agents->initAgents(data.acquireAgents(),
                locationMapping,
                agentTypeMapping,
                data.getAgentTypeLocTypes(),
                data.acquireProgressionMatrices());
            RandomGenerator::resize(agents->PPValues.size());
            std::cout << header << "H\tT\tP1\tP2\tQ\tQT\tNQ" << '\n';
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
        refreshAndPrintStatistics(simTime);
        while (simTime < endOfSimulation) {
            if (simTime.isMidnight()) {
                MovementPolicy<Simulation>::planLocations();
                if (simTime.getTimestamp() > 0) TestingPolicy<Simulation>::performTests(simTime, timeStep);
                if (simTime.getTimestamp() > 0) updateAgents(simTime);// No disease progression at launch
                if (enableOtherDisease) otherDisease(simTime,timeStep);
                refreshAndPrintStatistics(simTime);
            }
            MovementPolicy<Simulation>::movement(simTime, timeStep);
            InfectionPolicy<Simulation>::infectionsAtLocations(simTime, timeStep);
            ++simTime;
        }
        // thrust::copy(agents->agentStats.begin(), agents->agentStats.end(),
        // std::ostream_iterator<AgentStats>(std::cout, ""));
        agents->printAgentStatJSON(outAgentStat);
    }
};