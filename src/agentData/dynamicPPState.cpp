#include "dynamicPPState.h"

// static stuff
namespace detail {
    namespace DynamicPPState {
        unsigned h_numberOfStates = 0;
        char h_firstInfectedState = 0;
        char h_deadState;
        std::vector<float> h_infectious;
        std::vector<bool> h_susceptible;
        std::vector<bool> h_infected;
        std::vector<states::WBStates> h_WB;
        std::map<std::string, char> nameIndexMap;
        std::vector<ProgressionMatrix> h_transition;
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
        __constant__ unsigned numberOfStates = 0;
        __constant__ char firstInfectedState = 0;
        __constant__ float* infectious;
        __constant__ bool* susceptible;
        __constant__ bool* infected;
        __constant__ states::WBStates* WB;
        __constant__ char deadState;
        __constant__ ProgressionMatrix* d_transition;
#endif
    }// namespace DynamicPPState
}// namespace detail

HD ProgressionMatrix& DynamicPPState::getTransition() {
#ifdef __CUDA_ARCH__
    return *detail::DynamicPPState::d_transition;
#else
    return *detail::DynamicPPState::h_transition.data();
#endif
}

void HD DynamicPPState::updateMeta() {
#ifdef __CUDA_ARCH__
    infectious = detail::DynamicPPState::infectious[state];
    susceptible = detail::DynamicPPState::susceptible[state];
#else
    infectious = detail::DynamicPPState::h_infectious[state];
    susceptible = detail::DynamicPPState::h_susceptible[state];
#endif
}

std::string DynamicPPState::initTransitionMatrix(
    std::map<ProgressionType, std::pair<parser::TransitionFormat, unsigned>, std::less<>>&
        inputData,
    parser::ProgressionDirectory& config) {
    // init global parameters that are used to be static
    detail::DynamicPPState::h_numberOfStates = config.stateInformation.stateNames.size();
    detail::DynamicPPState::h_infectious =
        decltype(detail::DynamicPPState::h_infectious)(detail::DynamicPPState::h_numberOfStates);
    detail::DynamicPPState::h_WB =
        decltype(detail::DynamicPPState::h_WB)(detail::DynamicPPState::h_numberOfStates);
    detail::DynamicPPState::h_susceptible = decltype(detail::DynamicPPState::h_susceptible)(
        detail::DynamicPPState::h_numberOfStates, false);
    detail::DynamicPPState::h_transition.reserve(inputData.size());

    char idx = 0;
    std::string header;
    for (const auto& e : config.stateInformation.stateNames) {
        header += e + '\t';
        detail::DynamicPPState::nameIndexMap.emplace(std::make_pair(e, idx));
        ++idx;
    }

    for (const auto& e : config.states) {
        auto idx = detail::DynamicPPState::nameIndexMap.at(e.name);
        detail::DynamicPPState::h_WB[idx] = states::parseWBState(e.WB);
        detail::DynamicPPState::h_infectious[idx] = e.infectious;
    }

    for (unsigned i = 0; i < inputData.size(); ++i) {
        auto it = std::find_if(inputData.begin(), inputData.end(), [i](const auto& e) {
            return e.second.second == i;
        });
        detail::DynamicPPState::h_transition.emplace_back(it->second.first);
    }

    for (unsigned i = 0; i < detail::DynamicPPState::h_numberOfStates; i++) {
        if (detail::DynamicPPState::h_WB[i] == states::WBStates::D) {
            detail::DynamicPPState::h_deadState = i;
            break;
        }
    }


/*
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
            if (currentIt == inputData.states.end()) {
                throw IOProgression::MissingStateName(stateName);
            }
            std::iter_swap(it, currentIt);
        }
    }

    std::string stateName = inputData.firstInfectedState;
    auto currentIt = std::find_if(inputData.states.begin(),
        inputData.states.end(),
        [&stateName](const auto& s) { return s.stateName == stateName; });
    if (currentIt == inputData.states.end()) { throw IOProgression::MissingStateName(stateName); }
    detail::DynamicPPState::h_firstInfectedState =
        std::distance(inputData.states.begin(), currentIt);

    // setup name index mapping for the constructor
    std::string header;
    char idx = 0;
    for (const auto& s : inputData.states) {
        header += s.stateName + "\t";
        detail::DynamicPPState::h_infectious[idx] = s.infectious;
        detail::DynamicPPState::h_WB[idx] = states::parseWBState(s.WB);
        detail::DynamicPPState::nameIndexMap.emplace(std::make_pair(s.stateName, idx));
        detail::DynamicPPState::h_susceptible[idx] = (std::find(inputData.susceptibleStates.begin(),
                                                          inputData.susceptibleStates.end(),
                                                          s.stateName)
                                                      != inputData.susceptibleStates.end());
        ++idx;
    }
    header.pop_back();

    detail::DynamicPPState::transition = new ProgressionMatrix(inputData);
*/
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
    for(const auto& e : detail::DynamicPPState::h_transition) {
        
    }
    ProgressionMatrix* tmp = detail::DynamicPPState::transition->upload();
    cudaMemcpyToSymbol(detail::DynamicPPState::transition_gpu, &tmp, sizeof(ProgressionMatrix*));

    // do I have to make a fancy copy or it's automatic
    float* infTMP;
    cudaMalloc((void**)&infTMP, detail::DynamicPPState::h_numberOfStates * sizeof(float));
    cudaMemcpy(infTMP,
        detail::DynamicPPState::h_infectious,
        detail::DynamicPPState::h_numberOfStates * sizeof(float),
        cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(detail::DynamicPPState::infectious, &infTMP, sizeof(float*));

    states::SIRD* wbTMP;
    cudaMalloc((void**)&wbTMP, detail::DynamicPPState::h_numberOfStates * sizeof(states::SIRD));
    cudaMemcpy(wbTMP,
        detail::DynamicPPState::h_WB,
        detail::DynamicPPState::h_numberOfStates * sizeof(states::SIRD),
        cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(detail::DynamicPPState::WB, &wbTMP, sizeof(states::SIRD*));


    bool* susTMP;
    cudaMalloc((void**)&susTMP, detail::DynamicPPState::h_numberOfStates * sizeof(bool));
    cudaMemcpy(susTMP,
        detail::DynamicPPState::h_susceptible,
        detail::DynamicPPState::h_numberOfStates * sizeof(bool),
        cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(detail::DynamicPPState::susceptible, &susTMP, sizeof(bool*));

    cudaMemcpyToSymbol(detail::DynamicPPState::firstInfectedState,
        &detail::DynamicPPState::h_firstInfectedState,
        sizeof(detail::DynamicPPState::h_firstInfectedState));
    cudaMemcpyToSymbol(detail::DynamicPPState::firstInfectedState,
        &detail::DynamicPPState::h_firstInfectedState,
        sizeof(detail::DynamicPPState::h_firstInfectedState));
    cudaMemcpyToSymbol(detail::DynamicPPState::deadState,
        &detail::DynamicPPState::h_deadState,
        sizeof(detail::DynamicPPState::h_deadState));
#endif
    return header;
}

HD unsigned DynamicPPState::getNumberOfStates() {
#ifdef __CUDA_ARCH__
    return detail::DynamicPPState::numberOfStates;
#else
    return detail::DynamicPPState::h_numberOfStates;
#endif
}

std::vector<std::string> DynamicPPState::getStateNames() {
    std::vector<std::string> names(detail::DynamicPPState::h_numberOfStates);
    for (const auto& e : detail::DynamicPPState::nameIndexMap) { names[e.second] = e.first; }
    return names;
}

DynamicPPState::DynamicPPState(const std::string& name, unsigned progressionID_p)
    : progressionID(progressionID_p),
      state(detail::DynamicPPState::nameIndexMap.find(name)->second),
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

bool HD DynamicPPState::update(float scalingSymptons,
    AgentStats& stats,
    unsigned simTime,
    unsigned agentID,
    unsigned tracked) {
    if (daysBeforeNextState == -2) {
        daysBeforeNextState = getTransition().calculateJustDays(state);
    }
    if (daysBeforeNextState > 0) { --daysBeforeNextState; }
    if (daysBeforeNextState == 0) {
        states::WBStates oldWBState = this->getWBState();
        auto oldState = state;
        auto tmp = getTransition().calculateNextState(state, scalingSymptons);
        state = thrust::get<0>(tmp);
        updateMeta();
        daysBeforeNextState = thrust::get<1>(tmp);
        if (thrust::get<2>(tmp)) {// was a bad progression
            stats.worstState = state;
            stats.worstStateTimestamp = simTime;
            if (agentID == tracked) {
                printf(
                    "Agent %d bad progression %d->%d WBState: %d->%d for %d "
                    "days\n",
                    agentID,
                    oldState,
                    state,
                    oldWBState,
                    this->getWBState(),
                    daysBeforeNextState);
            }
        } else {// if (oldWBState != states::WBStates::W) this will record any
                // good progression!
            stats.worstStateEndTimestamp = simTime;
            if (agentID == tracked) {
                printf("Agent %d good progression %d->%d WBState: %d->%d\n",
                    agentID,
                    oldState,
                    state,
                    oldWBState,
                    this->getWBState());
            }
            if (this->getWBState() == states::WBStates::W) {// TODO: need isInfected() function!!
                return true;// recovered
            }
        }
    }
    return false;
}

states::WBStates DynamicPPState::getWBState() const {
#ifdef __CUDA_ARCH__
    return detail::DynamicPPState::WB[state];
#else
    return detail::DynamicPPState::h_WB[state];
#endif
}

HD char DynamicPPState::die() {
    daysBeforeNextState = -1;
#ifdef __CUDA_ARCH__
    state = detail::DynamicPPState::deadState;
    updateMeta();
    return detail::DynamicPPState::deadState;
#else
    state = detail::DynamicPPState::h_deadState;
    updateMeta();
    return detail::DynamicPPState::h_deadState;
#endif
}
