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
                0.000217643, 0.000108821, 0.000845925, 0.00267109, 0.000816392, 0.000958105, 0.00085024, //m general hospitalization
                0.000116508, 5.8254E-05, 0.000452839, 0.00083018, 0.000361234, 0.000542418, 0.000785243, //f      
                0.008400475, 0.004200238, 0.012600713, 0.027562562, 0.031510735, 0.036980507, 0.032817199, //m        death
                0.004496926, 0.002248463, 0.006745389, 0.0085665, 0.013942758, 0.020936024, 0.030308462, //f
                0.000446628, 0.000223314, 0.001708865, 0.005379398, 0.001675332, 0.001966144, 0.001744793, //m cardiov hospitalization
                0.00012258, 6.12901E-05, 0.000461947, 0.000841747, 0.000380061, 0.000570688, 0.000826168, //f
                0.003668178, 0.001834089, 0.005502266, 0.012035553, 0.013759575, 0.016148022, 0.014330059, //m        death
                0.001963642, 0.000981821, 0.002945462, 0.003740675, 0.006088288, 0.009141989, 0.013234587, //f
                0.000443433, 0.000221716, 0.001704071, 0.005368913, 0.001663345, 0.001952076, 0.001732309, //m pulmon  hospitalization
                0.00012087, 6.04348E-05, 0.000459381, 0.000838489, 0.000374757, 0.000562724, 0.000814638, //f
                0.016281963, 0.008140982, 0.024422945, 0.053422289, 0.061074715, 0.071676333, 0.063606928, //m        death
                0.008716029, 0.004358015, 0.013074044, 0.016603756, 0.027024123, 0.040578606, 0.058744446};//f
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

                
                if (RandomGenerator::randomReal(1.0) < suddenDeathProbs[!sex*7+ageGroup] && false) {
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