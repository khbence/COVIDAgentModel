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
    constexpr unsigned lengthInWeeks = 6;
    constexpr unsigned timeStep = 10;
    RandomGenerator::init(omp_get_max_threads());
    Simulation<PositionType,
        TypeOfLocation,
        PPStateSIRextended,
        BasicAgentMeta,
        NoMovement,
        BasicInfection>
        s;

    // setup for test
    {
        constexpr unsigned numAgents = 100000;
        constexpr double initial_infected_ratio = 0.01;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);

        // Create basic location for everyone
        s.addLocation(0, 0);

        // Populate agent list
        for (int i = 0; i < numAgents; i++) {
            s.addAgent(PPStateSIRextended(
                           dis(gen) < initial_infected_ratio ? states::SIRD::I : states::SIRD::S),
                false,
                0);
        }
    }

    s.initialization();
    s.runSimulation(timeStep, lengthInWeeks);
    END_PROFILING("main");
    Timing::report();
    return EXIT_SUCCESS;
}
