#include <iostream>
#include "simulation.h"
#include "movementPolicies.h"
#include "infectionPolicies.h"
#include "agentMeta.h"
#include "PPStateTypes.h"
// for testing
#include <inputJSON.h>
#include <random>
#include "randomGenerator.h"
#include <omp.h>
#include "timing.h"


using PositionType = int;
using TypeOfLocation = int;

int main(int argc, char const* argv[]) {
    BEGIN_PROFILING("main");
    constexpr unsigned lengthInWeeks = 12;
    constexpr unsigned timeStep = 10;
    RandomGenerator::init(omp_get_max_threads());
    Simulation<PositionType,
        TypeOfLocation,
        PPStateSIRextended,
        BasicAgentMeta,
        DummyMovement,
        BasicInfection>
        s;

    // setup for test
    {
        constexpr unsigned numAgents = 100000;
        constexpr double initial_infected_ratio = 0.05;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);

        // Create basic locations for everyone
        constexpr unsigned agentsPerLoc = 100;
        constexpr unsigned numLocations = numAgents / agentsPerLoc;
        for (int i = 0; i < numLocations; i++) s.addLocation(i, 0);

        // Populate agent list
        for (int i = 0; i < numAgents; i++) {
            s.addAgent(PPStateSIRextended(
                           dis(gen) < initial_infected_ratio ? states::SIRD::I : states::SIRD::S),
                false,
                i / agentsPerLoc);
        }
    }

    s.initialization();
    s.runSimulation(timeStep, lengthInWeeks);
    END_PROFILING("main");
    Timing::report();
    return EXIT_SUCCESS;
}
