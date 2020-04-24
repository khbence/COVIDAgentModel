#include <iostream>
#include "simulation.h"
#include "movementPolicies.h"
#include "infectionPolicies.h"
#include "diseaseProgressionPolicies.h"
#include "transitionMatrix.h"
#include "agentMeta.h"
// for testing
#include <inputJSON.h>
#include <random>
#include "randomGenerator.h"

using PositionType = int;
using TypeOfLocation = int;

int main(int argc, char const* argv[]) {
    constexpr unsigned lengthInWeeks = 10;
    constexpr unsigned timeStep = 10;
    RandomGenerator::init(1);
    Simulation<PositionType,
        TypeOfLocation,
        BasicAgentMeta,
        NoMovement,
        BasicInfection,
        ExtendedProgression>
        s;

    // setup for test
    {
        constexpr unsigned numAgents = 10000;
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
    return EXIT_SUCCESS;
}
