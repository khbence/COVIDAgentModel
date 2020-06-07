#include "dynamicPPState.h"
#include <vector>
#include <map>

// static stuff
namespace detail {
    namespace DynamicPPState {
        unsigned h_numberOfStates = 0;
        char h_firstInfectedState = 0;
        bool* h_infectious;
        bool* h_susceptible;
        states::WBStates* h_WB;
        std::map<std::string, char> nameIndexMap;
        SingleBadTransitionMatrix* transition;
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
        __device__ unsigned numberOfStates = 0;
        __device__ char firstInfectedState = 0;
        __device__ bool* infectious;
        __device__ bool* susceptible;
        __device__ states::WBStates* WB;
        __device__ SingleBadTransitionMatrix* transition_gpu;
#endif
    }// namespace DynamicPPState
}// namespace detail

HD SingleBadTransitionMatrix& DynamicPPState::getTransition() {
#ifdef __CUDA_ARCH__
    return *detail::DynamicPPState::transition_gpu;
#else
    return *detail::DynamicPPState::transition;
#endif
}

void HD DynamicPPState::updateMeta() {
#ifdef __CUDA_ARCH__
    infectious = detail::DynamicPPState::infectious[state];
#else
    infectious = detail::DynamicPPState::h_infectious[state];
#endif
#ifdef __CUDA_ARCH__
    susceptible = detail::DynamicPPState::susceptible[state];
#else
    susceptible = detail::DynamicPPState::h_susceptible[state];
#endif
}

void DynamicPPState::initTransitionMatrix(const std::string& inputFile) {
    auto inputData = DECODE_JSON_FILE(inputFile);

    // init global parameters that are used to be static
    detail::DynamicPPState::h_numberOfStates = inputData.states.size();
    detail::DynamicPPState::h_infectious = new bool[detail::DynamicPPState::h_numberOfStates];
    detail::DynamicPPState::h_WB = new states::WBStates[detail::DynamicPPState::h_numberOfStates];
    detail::DynamicPPState::h_susceptible = new bool[detail::DynamicPPState::h_numberOfStates];

    // state name and its occurence
    std::vector<std::pair<char, char>> mainStates{};
    for (const auto& s : inputData.states) {
        char sChar = s.stateName.front();
        auto it = std::find_if(mainStates.begin(), mainStates.end(), [sChar](const auto& current) {
            return current.first == sChar;
        });
        if (it == mainStates.end()) {
            mainStates.emplace_back(std::make_pair(sChar, 1));
        } else {
            ++it->second;
        }
    }

    // make sure that the states are not mixed
    auto it = inputData.states.begin();
    for (const auto& s : mainStates) {
        // the file indexes from 1
        for (char i = 1; i <= s.second; ++i, ++it) {
            std::string stateName;
            stateName.push_back(s.first);
            if (s.second > 1) { stateName += std::to_string(i); }
            auto currentIt = std::find_if(it, inputData.states.end(), [stateName](const auto& s) {
                return s.stateName == stateName;
            });
            if (currentIt == inputData.states.end()) { throw MissingStateName(stateName); }
            std::iter_swap(it, currentIt);
        }
    }

    std::string stateName = inputData.firstInfectedState;
    auto currentIt = std::find_if(inputData.states.begin(),
        inputData.states.end(),
        [&stateName](const auto& s) { return s.stateName == stateName; });
    if (currentIt == inputData.states.end()) { throw MissingStateName(stateName); }
    detail::DynamicPPState::h_firstInfectedState =
        std::distance(inputData.states.begin(), currentIt);

    // setup name index mapping for the constructor
    char idx = 0;
    for (const auto& s : inputData.states) {
        detail::DynamicPPState::h_infectious[idx] = s.infectious;
        detail::DynamicPPState::h_WB[idx] = states::parseWBState(s.WB);
        detail::DynamicPPState::nameIndexMap.emplace(std::make_pair(s.stateName, idx));
        detail::DynamicPPState::h_susceptible[idx] = (std::find(inputData.susceptibleStates.begin(),
                                                          inputData.susceptibleStates.end(),
                                                          s.stateName)
                                                      != inputData.susceptibleStates.end());
        ++idx;
    }

    detail::DynamicPPState::transition = new SingleBadTransitionMatrix(inputData);

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
    SingleBadTransitionMatrix* tmp = detail::DynamicPPState::transition->upload();
    cudaMemcpyToSymbol(
        detail::DynamicPPState::transition_gpu, &tmp, sizeof(SingleBadTransitionMatrix*));

    // do I have to make a fancy copy or it's automatic
    detail::DynamicPPState::numberOfStates = detail::DynamicPPState::h_numberOfStates;
    detail::DynamicPPState::firstInfectedState = detail::DynamicPPState::h_firstInfectedState;
    cudaMemcpyToSymbol(detail::DynamicPPState::infectious,
        detail::DynamicPPState::h_infectious,
        detail::DynamicPPState::h_numberOfStates * sizeof(*detail::DynamicPPState::h_infectious));
    cudaMemcpyToSymbol(detail::DynamicPPState::WB,
        detail::DynamicPPState::h_WB,
        detail::DynamicPPState::h_numberOfStates * sizeof(*detail::DynamicPPState::h_WB));
    cudaMemcpyToSymbol(detail::DynamicPPState::susceptible,
        detail::DynamicPPState::h_susceptible,
        detail::DynamicPPState::h_numberOfStates * sizeof(*detail::DynamicPPState::h_WB));
#endif
}

HD unsigned DynamicPPState::getNumberOfStates() {
#ifdef __CUDA_ARCH__
    return detail::DynamicPPState::numberOfStates;
#else
    return detail::DynamicPPState::h_numberOfStates;
#endif
}

DynamicPPState::DynamicPPState(const std::string& name)
    : state(detail::DynamicPPState::nameIndexMap.find(name)->second),
      daysBeforeNextState(getTransition().calculateJustDays(state)) {
    updateMeta();
}

void HD DynamicPPState::gotInfected() {
#ifdef __CUDA_ARCH__
    state = detail::DynamicPPState::firstInfectedState;
#else
    state = detail::DynamicPPState::h_firstInfectedState;
#endif
    daysBeforeNextState = -2;
    updateMeta();
}

void HD DynamicPPState::update(float scalingSymptons) {
    if (daysBeforeNextState == -2) {
        daysBeforeNextState = getTransition().calculateJustDays(state);
    }
    if (daysBeforeNextState > 0) { --daysBeforeNextState; }
    if (daysBeforeNextState == 0) {
        auto tmp = getTransition().calculateNextState(state, scalingSymptons);
        state = tmp.first;
        updateMeta();
        daysBeforeNextState = tmp.second;
    }
}