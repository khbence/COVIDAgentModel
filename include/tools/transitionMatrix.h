#pragma once
#include <array>
#include "optional.h"
#include <vector>
#include <string>
#include "progressionMatrixFormat.h"
#include "customExceptions.h"
#include <algorithm>
#include "randomGenerator.h"

class SingleBadTransitionMatrix {
    class NextStates {
        bool hasBad;
        thrust::pair<unsigned, float> bad;
        unsigned neutralCount;

        public:
        thrust::pair<unsigned, float> *neutral;
        NextStates(bool _hasBad, thrust::pair<unsigned, float> _bad, 
                    thrust::pair<unsigned, float>* _neutral, unsigned _neutralCount) :
            hasBad(_hasBad), bad(_bad), neutral(_neutral), neutralCount(_neutralCount) {}
        
        
        [[nodiscard]] unsigned selectNext(float scalingSypmtons) const {
            double random = RandomGenerator::randomUnit();
            double iterator = 0.0;
            double remainders = 0.0;
            if (hasBad) {
                iterator = bad.second * scalingSypmtons;
                if (random < iterator) { return bad.first; }
                remainders = (bad.second - iterator) / neutralCount;
            }
            unsigned idx = 0;
            do {
                iterator += neutral[idx].second + remainders;
                ++idx;
            } while (iterator < random);
            idx--;
            return neutral[idx].first;
        }
    };
    class NextStatesInit {
        public:
        // pair<index of new state,  raw chance to get there>
        stc::optional<std::pair<unsigned, float>> bad;
        std::vector<std::pair<unsigned, float>> neutral;

    
        NextStatesInit() = default;

        void addBad(std::pair<unsigned, float> bad_p) {
            if (bad) { throw(TooMuchBad(bad_p.first)); }
            bad = bad_p;
        }

        void addNeutral(std::pair<unsigned, float> newNeutral) { neutral.push_back(newNeutral); }

        void cleanUp(unsigned ownIndex) {
            if (neutral.empty()) {
                if (bad) {
                    neutral.push_back(bad.value());
                    bad.reset();
                } else {
                    neutral.emplace_back(ownIndex, 1.0F);
                }
            }
        }
    };

    class LengthOfState {
        int avgLength;
        int maxLength;
        double p;

    public:
        LengthOfState() = default;

        LengthOfState(int avgLength_p, int maxLength_p)
            : avgLength(avgLength_p),
              maxLength(maxLength_p),
              p(1.0 / static_cast<double>(avgLength_p)) {
            if (maxLength == -1) { maxLength = std::numeric_limits<decltype(maxLength)>::max(); }
        }

        // Note: [0, maxLength), because the 0 will run for a day, so the maxLength would run for
        // maxLength+1 days
        [[nodiscard]] int calculateDays() const {
            if (avgLength == -1) { return -1; }
            int days = RandomGenerator::geometric(p);
            while (maxLength < days) { days = RandomGenerator::geometric(p); }
            return days;
        }
    };

    NextStates *transitions;
    LengthOfState *lengths;
    unsigned numStates;


public:
    SingleBadTransitionMatrix() = default;
    explicit SingleBadTransitionMatrix(const std::string& fileName) {
        const auto inputData = DECODE_JSON_FILE(fileName);
        std::vector<NextStatesInit> initTransitions(inputData.states.size());
        numStates = inputData.states.size();
        lengths = (LengthOfState*)malloc(sizeof(LengthOfState)*inputData.states.size());
        transitions = (NextStates*)malloc(sizeof(NextStates)*inputData.states.size());
        
        auto getStateIndex = [&inputData](const std::string& name) {
            unsigned idx = 0;
            while (inputData.states[idx].stateName != name && idx < inputData.states.size()) {
                ++idx;
            }
            if (idx == inputData.states.size()) { throw(WrongStateName(name)); }
            return idx;
        };

        unsigned i = 0;
        for (const auto& s : inputData.states) {
            lengths[i] = LengthOfState{ s.avgLength, s.maxlength };
            double sumChance = 0.0;
            for (const auto& t : s.progressions) {
                auto idx = getStateIndex(t.name);
                sumChance += t.chance;
                if (t.isBadProgression) {
                    initTransitions[i].addBad(std::make_pair(idx, t.chance));
                } else {
                    initTransitions[i].addNeutral(std::make_pair(idx, t.chance));
                }
            }
            initTransitions[i].cleanUp(i);
            if (sumChance != 1.0 && !s.progressions.empty()) { throw(BadChances(s.stateName)); }
            thrust::pair<unsigned, float> badVal = initTransitions[i].bad ? initTransitions[i].bad.value() : thrust::pair<unsigned, float>(0,0.0f);
            thrust::pair<unsigned, float> *neutrals = (thrust::pair<unsigned, float> *)malloc(initTransitions[i].neutral.size()*sizeof(thrust::pair<unsigned, float>));
            for (int j = 0; j < initTransitions[i].neutral.size(); j++)
                neutrals[j] = initTransitions[i].neutral[j];
            transitions[i] = NextStates(initTransitions[i].bad?true:false, badVal, neutrals,initTransitions[i].neutral.size());
            ++i;
        }
    }

    ~SingleBadTransitionMatrix() {
        for (unsigned i = 0; i < numStates; i++) {
            free(transitions[i].neutral);
        }
        free(transitions);
        free(lengths);
    }

    [[nodiscard]] thrust::pair<unsigned, int> calculateNextState(unsigned currentState,
        float scalingSymptons) const {
        unsigned nextState = transitions[currentState].selectNext(scalingSymptons);
        int days = lengths[nextState].calculateDays();
        return thrust::make_pair(nextState, days);
    }

    [[nodiscard]] int calculateJustDays(unsigned state) const {
        return lengths[state].calculateDays();
    }
};