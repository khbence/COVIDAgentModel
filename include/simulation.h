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

                if (ppstate.getWBState() == states::WBStates::D) return;

                //
                // Parameters - to be extracted to config
                //
                //0-4ig	     5-14ig	      15-29ig	   30-59ig	    60-69ig	     70-79ig	  80+
                double randomHospProbs[] = {
                0.000326464, 0.000272053, 0.0011279, 0.002913917, 0.000933019, 0.001011333,    0.00085024, //m general hospitalization
                0.000174762, 0.000145635, 0.000603785, 0.000905651, 0.000412839, 0.000572553, 0.000785243, //f      
                0.012600713, 0.010500594, 0.016800951, 0.030068249, 0.036012268, 0.03903498,  0.032817199, //m        death
                0.006745389, 0.005621158, 0.008993852, 0.009345273, 0.015934581, 0.022099137, 0.030308462, //f
                1.70144E-05, 1.41787E-05, 2.26859E-05, 4.06004E-05, 4.86264E-05, 5.27079E-05, 4.43122E-05, //m cardiov hospitalization
                9.10813E-06, 7.59011E-06, 1.21442E-05, 1.26187E-05, 2.15161E-05, 2.98399E-05, 4.09247E-05, //f
                0.005502266, 0.004585222, 0.007336355, 0.013129695, 0.015725228, 0.017045135, 0.014330059, //m        death
                0.002945462, 0.002454552, 0.003927283, 0.004080736, 0.006958043, 0.009649877, 0.013234587, //f
                1.22211E-05, 1.01843E-05, 1.62948E-05, 2.91625E-05, 3.49274E-05, 3.78591E-05, 3.18286E-05, //m pulmon  hospitalization
                6.54219E-06, 5.45182E-06, 8.72292E-06, 9.06375E-06, 1.54546E-05, 2.14334E-05, 2.93954E-05, //f
                0.024422945, 0.020352454, 0.032563927, 0.058278861, 0.069799674, 0.075658351, 0.063606928, //m        death
                0.013074044, 0.010895036, 0.017432058, 0.018113189, 0.030884712, 0.042832973, 0.058744446};//f
                double avgLengths[] = {5.55, 2.78, 5.24};
                //0-4ig	     5-14ig	      15-29ig	   30-59ig	    60-69ig	     70-79ig	  80+
                double suddenDeathProbs[] = {
                    3.79825E-07, 3.79825E-07, 3.79825E-07, 3.84118E-06, 2.00505E-05, 3.47985E-05, 0.000105441,
                    2.03327E-07, 2.03327E-07, 2.03327E-07, 1.19385E-06, 8.87187E-06, 1.97007E-05, 9.73804E-05 }; //female
                

                uint8_t age = meta.getAge();
                uint8_t ageGroup = 0;
                if (age < 5) {
                    ageGroup = 0;
                } else if (age < 15) {
                    ageGroup = 1;
                } else if (age < 30) {
                    ageGroup = 2;
                } else if (age < 60) {
                    ageGroup = 3;
                } else if (age < 70) {
                    ageGroup = 4;
                } else if (age < 80) {
                    ageGroup = 5;
                } else {
                    ageGroup = 6;
                }
                bool sex = meta.getSex();
                //precond - 2 is cardiovascular, 4 is pulmonary. All others are general
                uint8_t type = meta.getPrecondIdx() == 2 ? 1 : (meta.getPrecondIdx() == 4 ? 2 : 0);

                //
                // non-COVID hospitalization - ended, see if dies or lives
                //
                if (timestamp>0 && agentStat.hospitalizedUntilTimestamp == timestamp) {
                    if (RandomGenerator::randomReal(1.0) < randomHospProbs[type * 4 * 7 + 2 * 7 + !sex * 7 + ageGroup]) {
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

                //If already dead, or in hospital (due to COVID or non-COVID), return
                if (ppstate.getWBState() == states::WBStates::S ||
                    timestamp < agentStat.hospitalizedUntilTimestamp) return;

                
                if (RandomGenerator::randomReal(1.0) < suddenDeathProbs[!sex*7+ageGroup]) {
                    agentStat.worstState = ppstate.die(false); //not COVID-related
                    agentStat.worstStateTimestamp = timestamp;
                    if (agentID == tracked) {
                        printf("Agent %d (%s, age %d) died of sudden death, timestamp %d\n", tracked, sex?"M":"F", (int)age, timestamp);
                    }
                    return;
                }

                //
                // Random hospitalization
                //
                double probability = randomHospProbs[type * 4 * 7 + !sex * 7 + ageGroup];
                if (RandomGenerator::randomReal(1.0) < probability) {
                    //Got hospitalized
                    //Length;
                    unsigned avgLength = avgLengths[type]; //In days
                    double p = 1.0 / (double) avgLength;
                    unsigned length = RandomGenerator::geometric(p);
                    if (length == 0) length = 1; // At least one day
                    agentStat.hospitalizedTimestamp = timestamp;
                    agentStat.hospitalizedUntilTimestamp = timestamp + length*24*60/timeStep;
                    //printf("Agent %d hospitalized for non-COVID disease, timestamp %d-%d\n", agentID, timestamp, agentStat.hospitalizedUntilTimestamp);
                    if (agentID == tracked) {
                        printf("Agent %d (%s, age %d) hospitalized for non-COVID disease, timestamp %d-%d\n", agentID, sex?"M":"F", (int)age, timestamp, agentStat.hospitalizedUntilTimestamp);
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
                data.acquireProgressionMatrices(),
                data.acquireLocationTypes());
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