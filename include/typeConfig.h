#pragma once
#include <array>
#include "progressionMatrices.h"
#include "simulation.h"
#include "dynamicPPState.h"
#include "movementPolicies.h"
#include "infectionPolicies.h"

namespace types {
    using PositionType = std::array<double, 2>;
    using TypeOfLocation = unsigned;
    using ProgressionMatrix = MultiBadMatrix;
    using PPStates = DynamicPPState<ProgressionMatrix>;
    using Simulation_t = Simulation<PositionType, TypeOfLocation, PPStates, BasicAgentMeta, DummyMovement, BasicInfection>;
}
