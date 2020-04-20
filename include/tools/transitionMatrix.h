#pragma once
#include <array>
#include <optional>
#include <vector>
#include <string>
#include "transitions.h"
#include "customExceptions.h"
#include <algorithm>

template<unsigned N>
class SingleBadTransitionMatrix {
    class NextStates {
        // pair<index of new state,  raw chance to get there>
        std::optional<std::pair<unsigned, float>> bad;
        std::vector<std::pair<unsigned, float>> neutral;

    public:
        NextStates() = default;

        void addBad(std::pair<unsigned, float> bad_p) {
            if (bad) { throw(TooMuchBad(bad_p.first)); }
            bad = bad_p;
        }

        void addNeutral(std::pair<unsigned, float> newNeutral) { neutral.push_back(newNeutral); }

        void cleanUp(unsigned ownIndex) {
            if (neutral.size() == 0) {
                if (bad) {
                    neutral.push_back(bad.value());
                    bad.reset();
                } else {
                    neutral.emplace_back(ownIndex, 1.0F);
                }
            }
        }

        [[nodiscard]] unsigned selectNext(float scalingSypmtons) const {
            double random = 0.8;// TODO create random number creator
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

    public:
        LengthOfState() = default;

        LengthOfState(int avgLength_p, int maxLength_p)
            : avgLength(avgLength_p), maxLength(maxLength_p) {}

        [[nodiscard]] int calculateDays() const {
            return avgLength;
        }// TODO make the nice distribution
    };

    std::array<NextStates, N> transitions;
    std::array<LengthOfState, N> lengths;


public:
    SingleBadTransitionMatrix() = default;
    explicit SingleBadTransitionMatrix(const std::string& fileName) {
        auto inputData = jsond::JSONDecodable<parser::TransitionFormat>::DecodeFromFile(fileName);
        if (inputData.states.size() != N) {
            throw(WrongNumberOfStates(N, inputData.states.size()));
        }
        auto getStateIndex = [&inputData](const std::string& name) {
            unsigned idx = 0;
            while (inputData.states[idx].stateName != name) { ++idx; }
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
            if (sumChance != 1.0) { throw(BadChances(s.stateName)); }
            ++i;
        }
    }

    [[nodiscard]] std::pair<unsigned, int> calculateNextState(unsigned currentState,
        float scalingSymptons) const {
        unsigned nextState = transitions[currentState].selectNext(scalingSymptons);
        unsigned days = lengths[nextState].calculateDays();
        return std::make_pair(nextState, days);
    }
};