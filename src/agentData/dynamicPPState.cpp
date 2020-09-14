#include "dynamicPPState.h"
#include <cassert>
#include "customExceptions.h"

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
        __constant__ ProgressionMatrix** d_transition;
#endif
    }// namespace DynamicPPState
}// namespace detail

HD ProgressionMatrix& DynamicPPState::getTransition(unsigned progressionID_p) {
#ifdef __CUDA_ARCH__
    return *detail::DynamicPPState::d_transition[progressionID_p];
#else
    return detail::DynamicPPState::h_transition[progressionID_p];
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
        auto idx = detail::DynamicPPState::nameIndexMap.at(e.stateName);
        detail::DynamicPPState::h_WB[idx] = states::parseWBState(e.WB);
        detail::DynamicPPState::h_infectious[idx] = e.infectious;
    }

    for (unsigned i = 0; i < inputData.size(); ++i) {
        auto it = std::find_if(inputData.begin(), inputData.end(), [i](const auto& e) {
            return e.second.second == i;
        });
        assert(it != inputData.end());
        detail::DynamicPPState::h_transition.emplace_back(it->second.first);
    }

    for (unsigned i = 0; i < detail::DynamicPPState::h_numberOfStates; i++) {
        if (detail::DynamicPPState::h_WB[i] == states::WBStates::D) {
            detail::DynamicPPState::h_deadState = i;
            break;
        }
    }

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
    ProgressionMatrix* tmpDevice;
    cudaMalloc((void**)&tmpDevice,
        detail::DynamicPPState::h_transition.size()
            * sizeof(decltype(detail::DynamicPPState::d_transition)));

    std::vector<ProgressionMatrix*> tmp;
    for (auto& e : detail::DynamicPPState::h_transition) { tmp.push_back(e.upload()); }
    cudaMemcpy(
        tmpDevice, tmp.data(), tmp.size() * sizeof(ProgressionMatrix*), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(
        detail::DynamicPPState::d_transition, &tmpDevice, sizeof(decltype(tmpDevice)));

    float* infTMP;
    cudaMalloc((void**)&infTMP, detail::DynamicPPState::h_numberOfStates * sizeof(float));
    cudaMemcpy(infTMP,
        detail::DynamicPPState::h_infectious.data(),
        detail::DynamicPPState::h_numberOfStates * sizeof(float),
        cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(detail::DynamicPPState::infectious, &infTMP, sizeof(float*));

    states::SIRD* wbTMP;
    cudaMalloc((void**)&wbTMP, detail::DynamicPPState::h_numberOfStates * sizeof(states::SIRD));
    cudaMemcpy(wbTMP,
        detail::DynamicPPState::h_WB.data(),
        detail::DynamicPPState::h_numberOfStates * sizeof(states::SIRD),
        cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(detail::DynamicPPState::WB, &wbTMP, sizeof(states::SIRD*));

    bool* tmpSusceptible = new bool[detail::DynamicPPState::h_susceptible.size()];
    std::copy(detail::DynamicPPState::h_susceptible.begin(),
        detail::DynamicPPState::h_susceptible.end(),
        tmpSusceptible);
    bool* susTMP;
    cudaMalloc((void**)&susTMP, detail::DynamicPPState::h_numberOfStates * sizeof(bool));
    cudaMemcpy(susTMP,
        tmpSusceptible,
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
      daysBeforeNextState(getTransition(progressionID).calculateJustDays(state)) {
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
        daysBeforeNextState = getTransition(progressionID).calculateJustDays(state);
    }
    if (daysBeforeNextState > 0) { --daysBeforeNextState; }
    if (daysBeforeNextState == 0) {
        states::WBStates oldWBState = this->getWBState();
        auto oldState = state;
        auto tmp = getTransition(progressionID).calculateNextState(state, scalingSymptons);
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
            if (this->isInfected() == false) {
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
bool HD DynamicPPState::isInfected() const {
#ifdef __CUDA_ARCH__
    return detail::DynamicPPState::infected[state];
#else
    return detail::DynamicPPState::h_infected[state];
#endif
}
