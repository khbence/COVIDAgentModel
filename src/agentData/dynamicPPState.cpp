#include "dynamicPPState.h"
#include <vector>

// static stuff
namespace detail {
    namespace DynamicPPState {
        unsigned h_numberOfStates = 0;
        char h_firstInfectedState = 0;
        bool* h_infectious;
        states::WBStates* h_WB;
        SingleBadTransitionMatrix* transition;
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
        __device__ unsigned numberOfStates = 0;
        __device__ char firstInfectedState = 0;
        __device__ bool* infectious;
        __device__ states::WBStates* WB;
        __device__ SingleBadTransitionMatrix* transition_gpu;
#endif
    }// namespace DynamicPPState
}// namespace detail

void DynamicPPState::initTransitionMatrix(const std::string& inputFile) {
    auto inputData = DECODE_JSON_FILE(inputFile);

    // init global parameters that are used to be static
    detail::DynamicPPState::h_numberOfStates = inputData.states.size();
    detail::DynamicPPState::h_infectious = new bool[detail::DynamicPPState::h_numberOfStates];
    detail::DynamicPPState::h_WB = new states::WBStates[detail::DynamicPPState::h_numberOfStates];

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
    unsigned idx = 0;
    for (const auto& s : mainStates) {
        // the file indexes from 1
        for (char i = 1; i <= s.second; ++i, ++it, ++idx) {
            std::string stateName{ std::to_string(s.first) };
            if (s.second > 1) { stateName += std::to_string(i); }
            auto currentIt = std::find_if(it, inputData.states.end(), [stateName](const auto& s) {
                return s.stateName == stateName;
            });
            if (currentIt == inputData.states.end()) { throw MissingStateName(stateName); }
            std::iter_swap(it, currentIt);
            detail::DynamicPPState::h_infectious[idx] = it->infectious;
            detail::DynamicPPState::h_WB[idx] = states::parseWBState(it->WB);
        }
    }

    std::string stateName = inputData.firstInfectedState;
    auto currentIt = std::find_if(it, inputData.states.end(), [&stateName](const auto& s) {
        return s.stateName == stateName;
    });


    detail::DynamicPPState::transition = new SingleBadTransitionMatrix(inputData);
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
    SingleBadTransitionMatrix* tmp = detail::DynamicPPState::transition->upload();
    cudaMemcpyToSymbol(
        detail::DynamicPPState::transition_gpu, &tmp, sizeof(SingleBadTransitionMatrix*));

    // do I have to make a fancy copy or it automatic
    detail::DynamicPPState::numberOfStates = detail::DynamicPPState::h_numberOfStates;
    cudaMemcpyToSymbol(detail::DynamicPPState::infectious,
        detail::DynamicPPState::h_infectious,
        detail::DynamicPPState::h_numberOfStates * sizeof(*detail::DynamicPPState::h_infectious));
    cudaMemcpyToSymbol(detail::DynamicPPState::WB,
        detail::DynamicPPState::h_WB,
        detail::DynamicPPState::h_numberOfStates * sizeof(*detail::DynamicPPState::h_WB));
#endif
}


HD DynamicPPState::DynamicPPState(const std::string& state) {}