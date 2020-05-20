#pragma once
#include <array>
#include "optional.h"
#include <vector>
#include <string>
#include "progressionMatrixFormat.h"
#include "customExceptions.h"
#include <algorithm>
#include "randomGenerator.h"
#define NSTATES 11
class SingleBadTransitionMatrix {
    class NextStates {
        // pair<index of new state,  raw chance to get there>
        stc::optional<std::pair<unsigned, float>> bad;
        std::vector<std::pair<unsigned, float>> neutral;

    public:
        NextStates() = default;

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

        [[nodiscard]] unsigned selectNext(float scalingSypmtons) const {
            double random = RandomGenerator::randomUnit();
            double iterator = 0.0;
            double remainders = 0.0;
            if (bad) {
                iterator = bad.value().second * scalingSypmtons;
                if (random < iterator) { return bad.value().first; }
                remainders = (bad.value().second - iterator) / neutral.size();
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

    std::vector<NextStates> transitions;
    std::vector<LengthOfState> lengths;


public:
    SingleBadTransitionMatrix() = default;
    explicit SingleBadTransitionMatrix(const std::string& fileName) {
        const auto inputData = DECODE_JSON_FILE(fileName);
        transitions.resize(inputData.states.size());
        lengths.resize(inputData.states.size());
        
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
                    transitions[i].addBad(std::make_pair(idx, t.chance));
                } else {
                    transitions[i].addNeutral(std::make_pair(idx, t.chance));
                }
            }
            transitions[i].cleanUp(i);
            if (sumChance != 1.0 && !s.progressions.empty()) { throw(BadChances(s.stateName)); }
            ++i;
        }
    }

    [[nodiscard]] std::pair<unsigned, int> calculateNextState(unsigned currentState,
        float scalingSymptons) const {
        unsigned nextState = transitions[currentState].selectNext(scalingSymptons);
        int days = lengths[nextState].calculateDays();
        return std::make_pair(nextState, days);
    }

    [[nodiscard]] int calculateJustDays(unsigned state) const {
        return lengths[state].calculateDays();
    }
};