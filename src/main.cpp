#include <iostream>
#include "PPStateSIRBasic.h"
#include "simulation.h"
#include "noMovement.h"

/*
template<class PPState, class LocationType>
void infectionAtLocation(LocationType& location,
    std::chrono::system_clock::time_point simClock,
    std::chrono::minutes timeStep,
    LocationStats& stats) {
    std::vector<unsigned>& agents = location.getAgents();
    unsigned sick = std::count_if(agents.begin(), agents.end(), [](unsigned i) {
        return AgentList<PPState, LocationType>::getInstance()->getPPState(i).getSIRD() == states::SIRD::I;
    });
    stats.sick = sick;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    std::for_each(agents.begin(), agents.end(), [&](unsigned i) {
        if (AgentList<PPState, LocationType>::getInstance()->getPPState(i).getSIRD() == states::SIRD::S && dis(gen) < 0.1) {
            AgentList<PPState, LocationType>::getInstance()->getPPState(i).gotInfected();
        }
    });
    unsigned sickAfter = std::count_if(agents.begin(), agents.end(), [](unsigned i) {
        return AgentList<PPState, LocationType>::getInstance()->getPPState(i).getSIRD() == states::SIRD::I;
    });
    stats.infected = sickAfter - sick;
}
*/

int main(int argc, char const* argv[]) {
    constexpr unsigned lengthInWeeks = 2;
    Simulation<PPStateSIRBasic, int, int, NoMovement> s;
    s.runSimulation(lengthInWeeks);
    return EXIT_SUCCESS;

    /*
        int numAgents = 1000;
        if (argc > 1) numAgents = atoi(argv[1]);
        double initial_infected_ratio = 0.01;
        if (argc > 1) initial_infected_ratio = atof(argv[2]);

        // Basic config - PPStateSIRBasic for agent states, int PositionType, int TypeOfLocation
        Simulation<PPStateSIRBasic, int, int> s;

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);

        // Create basic location for everyone
        s.locationList().push_back(Location<int, int>(0, 0));
        Location<int, int>* location = &(s.locationList()[0]);

        // Populate agent list
        for (int i = 0; i < numAgents; i++) {
            s.agentList()->addAgent(PPStateSIRBasic(dis(gen) < initial_infected_ratio ? states::SIRD::I : states::SIRD::S), false, location);
        }

        // Time now
        std::chrono::system_clock::time_point simClock = std::chrono::system_clock::now();
        // Time step
        std::chrono::minutes timeStep = std::chrono::minutes(10);
        // End time
        std::chrono::system_clock::time_point endTime = simClock + std::chrono::hours(24 * 7 * 2);// two weeks

        // Time loop
        while (simClock < endTime) {
            simClock += timeStep;
            LocationStats stats;
            infectionAtLocation<PPStateSIRBasic>(*location, simClock, timeStep, stats);
            std::time_t t_c = std::chrono::system_clock::to_time_t(simClock);
            std::cout << std::put_time(std::localtime(&t_c), "%F %T") << " Sick: " << stats.sick << " New Infections: " << stats.infected << '\n';
        }
    */
}
